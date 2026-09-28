"""Fetch and normalize Open-Meteo current, hourly, and daily forecast fields."""
import json
import math
import statistics
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "wind_direction_10m", "wind_speed_10m", "is_day", "wind_gusts_10m",
    "rain", "precipitation", "showers", "weather_code", "cloud_cover",
)
HOURLY_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation_probability", "precipitation", "rain", "showers",
    "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm",
    "soil_temperature_54cm", "soil_moisture_27_to_81cm",
    "soil_moisture_9_to_27cm", "soil_moisture_3_to_9cm",
    "soil_moisture_1_to_3cm", "soil_moisture_0_to_1cm", "weather_code",
    "surface_pressure", "pressure_msl", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration",
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m",
    "wind_direction_180m", "wind_gusts_10m", "temperature_80m",
    "temperature_120m", "temperature_180m",
)
DAILY_VARIABLES = (
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_max", "uv_index_max", "apparent_temperature_min",
    "uv_index_clear_sky_max", "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "sunrise", "sunset", "sunshine_duration",
    "moonrise", "daylight_duration", "moonset", "moon_phase", "rain_sum",
    "showers_sum", "precipitation_sum", "precipitation_probability_max",
    "precipitation_hours",
)


def _number(daily, variable, index):
    values = daily.get(variable)
    if not values or index >= len(values) or values[index] is None:
        raise ValueError(f"Missing ECMWF daily value: {variable} at index {index}")
    return float(values[index])


def _hourly_mean(hourly, variable, indexes):
    values = hourly.get(variable, [])
    samples = [float(values[index]) for index in indexes if index < len(values) and values[index] is not None]
    if not samples:
        raise ValueError(f"Missing Open-Meteo hourly values: {variable}")
    return statistics.mean(samples)


def _distance_km(first, second):
    lat1, lon1 = math.radians(first["lat"]), math.radians(first["lon"])
    lat2, lon2 = math.radians(second["lat"]), math.radians(second["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def fetch_live_nwp(regions, error_patterns):
    """Fetch all requested Open-Meteo fields and build Day 1–10 model features."""
    params = {
        "latitude": ",".join(str(region["lat"]) for region in regions),
        "longitude": ",".join(str(region["lon"]) for region in regions),
        "current": ",".join(CURRENT_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "forecast_days": 11,
    }
    request = Request(
        f"{OPEN_METEO_URL}?{urlencode(params)}",
        headers={"User-Agent": "Clime/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    provider_locations = payload if isinstance(payload, list) else [payload]
    if len(provider_locations) != len(regions):
        raise ValueError(f"Expected {len(regions)} Open-Meteo locations, received {len(provider_locations)}")

    normalized = []
    location_summaries = []
    details = []
    for region, location in zip(regions, provider_locations):
        daily = location.get("daily", {})
        hourly = location.get("hourly", {})
        dates = daily.get("time", [])
        hourly_times = hourly.get("time", [])
        if len(dates) < 11:
            raise ValueError(f"Open-Meteo returned fewer than 11 daily values for {region['name']}")

        hourly_indexes_by_date = {}
        for hourly_index, timestamp in enumerate(hourly_times):
            hourly_indexes_by_date.setdefault(timestamp[:10], []).append(hourly_index)

        daily_fields = {
            "time": dates[:11],
            **{variable: daily.get(variable, [])[:11] for variable in DAILY_VARIABLES},
        }
        grid_id = region.get("grid_id", f"IN-{region['name'][:2].upper()}")
        current = location.get("current", {})
        location_summaries.append({
            "region": region["name"],
            "grid_id": grid_id,
            "current": current,
            "current_units": location.get("current_units", {}),
            "daily": daily_fields,
            "daily_units": location.get("daily_units", {}),
        })
        details.append({
            "region": region["name"],
            "grid_id": grid_id,
            "current": current,
            "current_units": location.get("current_units", {}),
            "daily": daily_fields,
            "daily_units": location.get("daily_units", {}),
            "hourly": hourly,
            "hourly_units": location.get("hourly_units", {}),
        })

        history = error_patterns.get("regions", {}).get(region["name"], {}).get("by_lead_time", [])
        history_by_day = {row["lead_time"]: row for row in history}

        for lead_time in range(1, 11):
            index = lead_time
            date = dates[index]
            hourly_indexes = hourly_indexes_by_date.get(date, [])
            if not hourly_indexes:
                raise ValueError(f"Open-Meteo has no hourly values for {region['name']} on {date}")

            precipitation = _number(daily, "precipitation_sum", index)
            temperature = _hourly_mean(hourly, "temperature_2m", hourly_indexes)
            pressure = _hourly_mean(hourly, "pressure_msl", hourly_indexes)
            humidity = _hourly_mean(hourly, "relative_humidity_2m", hourly_indexes)
            wind_speed = _number(daily, "wind_speed_10m_max", index)
            wind_direction = _number(daily, "wind_direction_10m_dominant", index)
            recent_precip = [
                _number(daily, "precipitation_sum", day_index)
                for day_index in range(max(0, index - 2), index + 1)
            ]
            previous_date = dates[index - 1]
            previous_indexes = hourly_indexes_by_date.get(previous_date, [])
            previous_pressure = _hourly_mean(hourly, "pressure_msl", previous_indexes)
            historical = history_by_day.get(lead_time, {})

            normalized.append({
                "region": region["name"],
                "grid_id": grid_id,
                "lat": region["lat"],
                "lon": region["lon"],
                "latitude": region["lat"],
                "longitude": region["lon"],
                "day": lead_time,
                "lead_time": lead_time,
                "valid_date": date,
                "precipitation": precipitation,
                "temperature": temperature,
                "pressure": pressure,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "historical_error": float(historical.get("mean_historical_error_feature", 0)),
                "pressure_variation": abs(pressure - previous_pressure),
                "precipitation_variability": statistics.pstdev(recent_precip),
                "spatial_gradient": 0.0,
                "current_conditions": current,
            })

    by_day = {}
    for cell in normalized:
        by_day.setdefault(cell["day"], []).append(cell)
    region_by_name = {region["name"]: region for region in regions}
    for cells in by_day.values():
        for cell in cells:
            origin = region_by_name[cell["region"]]
            neighbors = sorted(
                (other for other in cells if other["region"] != cell["region"]),
                key=lambda other: _distance_km(origin, region_by_name[other["region"]]),
            )[:4]
            if neighbors:
                cell["spatial_gradient"] = statistics.mean(
                    abs(cell["precipitation"] - neighbor["precipitation"]) / 100
                    for neighbor in neighbors
                )

    return {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "forecast_start_date": provider_locations[0]["daily"]["time"][0],
        "provider": "Open-Meteo",
        "model": "Open-Meteo automatic model selection",
        "model_resolution": "Provider-selected by location",
        "locations": location_summaries,
        "details": details,
        "cells": normalized,
    }
