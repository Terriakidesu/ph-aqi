import pandas as pd
import constants
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import math
import numpy as np
import os

daily_df = pd.read_csv("datasets/selected/202601/202601_SelectedData.csv")
daily_df["datetime"] = pd.to_datetime(
    daily_df["datetime"]).dt.tz_localize(None)

start_date = "2026-01-01"
end_date = "2026-01-31"

regions = {city: [city] for city in constants.CITIES}


for region, cities in regions.items():

    for current_day in pd.date_range(start_date, end_date, freq="D"):
        next_day = current_day + pd.DateOffset(1)
        date_str = current_day.strftime("%b %d, %Y - %a")

        components = constants.COMPONENTS[:-1]
        num_components = len(components)

        fig, axes = plt.subplots(
            num_components, 1, figsize=(12, 4 * num_components))

        # Handle case where there's only one component
        if num_components == 1:
            axes = [axes]

        for idx, column_name in enumerate(components):
            ax = axes[idx]

            # column_name = comoponent
            name = constants.COMPONENT_NAMES[column_name]

            if len(cities) > 0:
                ax.set_title(
                    f"Graph for hourly levels of {name} in {region} - {date_str}")
            else:
                ax.set_title(
                    f"Graph for hourly levels of {name} in {city} - {date_str}")

            all_datetimes = []
            all_values = []

            for city in cities:
                city_data = daily_df[daily_df["city_name"] == city]

                current_day_data = city_data[city_data["datetime"].between(
                    current_day, next_day)]

                current_day_data["datetime"] = current_day_data["datetime"].dt.floor(
                    freq='h')

                ax.plot(current_day_data["datetime"],
                        current_day_data[column_name], label=city)

                all_datetimes.extend(current_day_data["datetime"])
                all_values.extend(current_day_data[column_name])

            # Set axis properties BEFORE trend line
            ax.legend()

            max_val = math.ceil(daily_df[column_name].max())
            divisions = min(10, max_val)
            steps = max_val // divisions

            ax.set_yticks(range(0, max_val + steps, steps))
            ax.set_xticks(pd.date_range(current_day, next_day, freq='h'))
            ax.xaxis.set_major_formatter(mdt.DateFormatter('%I:%M %p'))
            ax.tick_params(axis='x', rotation=45)

            # Save axis limits before plotting trend
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()

            # Plot trend line on top after axis setup
            if len(all_datetimes) > 1 and len(all_values) > 1:
                try:
                    trend_df = pd.DataFrame({
                        "datetime": pd.to_datetime(all_datetimes),
                        "value": pd.to_numeric(all_values, errors="coerce")
                    }).dropna()

                    if not trend_df.empty:
                        avg_by_hour = trend_df.groupby("datetime", as_index=False)[
                            "value"].mean()
                        avg_by_hour = avg_by_hour.sort_values("datetime")

                        if len(avg_by_hour) > 1:
                            # Fit trend line using hourly average across all cities
                            x_numeric = np.arange(len(avg_by_hour))
                            z = np.polyfit(x_numeric, avg_by_hour["value"], 1)
                            p = np.poly1d(z)
                            y_fit = p(x_numeric)

                            ax.plot(
                                avg_by_hour["datetime"],
                                y_fit,
                                'r-',
                                linewidth=3,
                                label='Trend (City Average)',
                                zorder=10
                            )
                            ax.legend()
                except:
                    pass

            # Force axis limits back to original
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

        plt.tight_layout()

        os.makedirs(f"figs/{region}", exist_ok=True)

        plt.savefig(f"figs/{region}/{region}_{date_str}.png")
        print(f"Done: figs/{region}/{region}_{date_str}.png")

        plt.close(fig)

    # plt.show()
