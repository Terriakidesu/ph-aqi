
CITIES = [
    "Manila",
    "Quezon City",
    "Makati City",
    "Pasig",
    "Cebu City",
    "Lapu-Lapu City",
    "Iloilo City",
    "Bacolod",
    "Davao",
    "Cagayan de Oro",
]

CITY_TO_REGION = {
    "Manila": "National Capital Region",
    "Quezon City": "National Capital Region",
    "Makati City": "National Capital Region",
    "Pasig": "National Capital Region",
    "Cebu City": "Central Visayas",
    "Lapu-Lapu City": "Central Visayas",
    "Iloilo City": "Western Visayas",
    "Bacolod": "Negros Island Region",
    "Davao": "Davao Region",
    "Cagayan de Oro": "Northern Mindanao",
}

REGION_TO_CITIES = {
    "National Capital Region": [
        "Manila",
        "Quezon City",
        "Makati City",
        "Pasig",
    ],
    "Central Visayas": ["Cebu City", "Lapu-Lapu City"],
    "Western Visayas": ["Iloilo City"],
    "Negros Island Region": ["Bacolod"],
    "Davao Region": ["Davao"],
    "Northern Mindanao": ["Cagayan de Oro"],
}

CITY_TO_ISLAND_GROUP = {
    "Manila": "Luzon",
    "Quezon City": "Luzon",
    "Makati City": "Luzon",
    "Pasig": "Luzon",
    "Cebu City": "Visayas",
    "Lapu-Lapu City": "Visayas",
    "Iloilo City": "Visayas",
    "Bacolod": "Visayas",
    "Davao": "Mindanao",
    "Cagayan de Oro": "Mindanao",
}

ISLAND_GROUP_TO_CITIES = {
    "Luzon": ["Manila", "Quezon City", "Makati City", "Pasig"],
    "Visayas": ["Cebu City", "Lapu-Lapu City", "Iloilo City", "Bacolod"],
    "Mindanao": ["Davao", "Cagayan de Oro"],
}

COMPONENT_NAMES = {
    "main.aqi": "Air Quality Index",
    "components.co": "Carbon Monoxide (CO)",
    "components.no": "Nitrogen Monoxide (NO)",
    "components.no2": "Nitrogen Dioxide (NO2)",
    "components.o3": "Ozone (O3)",
    "components.so2": "Sulfur Dioxide (SO2)",
    "components.pm2_5": "Particulate Matter 2.5 (PM2.5)",
    "components.pm10": "Particulate Matter 10 (PM10)",
    "components.nh3": "Ammonia (NH3)",
    "city_name": "City"
}

COMPONENTS = [
    "main.aqi",
    "components.co",
    "components.no",
    "components.no2",
    "components.o3",
    "components.so2",
    "components.pm2_5",
    "components.pm10",
    "components.nh3",
    "city_name"
]

UNITS = "μg/m³"  # microgram per cubic meter
