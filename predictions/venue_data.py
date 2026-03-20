#!/usr/bin/env python3
"""
Static venue data for NBA arenas.
Used by scout_venue() in NEXUS v4 for altitude and timezone adjustments.
Zero API calls — all hardcoded.
"""

VENUE_MAP = {
    'ATL': {'altitude': 1050, 'tz': 'ET', 'arena': 'State Farm Arena', 'lat': 33.757, 'lng': -84.396},
    'BOS': {'altitude': 141, 'tz': 'ET', 'arena': 'TD Garden', 'lat': 42.366, 'lng': -71.062},
    'BKN': {'altitude': 30, 'tz': 'ET', 'arena': 'Barclays Center', 'lat': 40.683, 'lng': -73.975},
    'CHA': {'altitude': 751, 'tz': 'ET', 'arena': 'Spectrum Center', 'lat': 35.225, 'lng': -80.839},
    'CHI': {'altitude': 594, 'tz': 'CT', 'arena': 'United Center', 'lat': 41.881, 'lng': -87.674},
    'CLE': {'altitude': 653, 'tz': 'ET', 'arena': 'Rocket Mortgage FieldHouse', 'lat': 41.497, 'lng': -81.688},
    'DAL': {'altitude': 430, 'tz': 'CT', 'arena': 'American Airlines Center', 'lat': 32.790, 'lng': -96.810},
    'DEN': {'altitude': 5280, 'tz': 'MT', 'arena': 'Ball Arena', 'lat': 39.749, 'lng': -105.008},
    'DET': {'altitude': 600, 'tz': 'ET', 'arena': 'Little Caesars Arena', 'lat': 42.341, 'lng': -83.055},
    'GSW': {'altitude': 7, 'tz': 'PT', 'arena': 'Chase Center', 'lat': 37.768, 'lng': -122.388},
    'HOU': {'altitude': 80, 'tz': 'CT', 'arena': 'Toyota Center', 'lat': 29.751, 'lng': -95.362},
    'IND': {'altitude': 715, 'tz': 'ET', 'arena': 'Gainbridge Fieldhouse', 'lat': 39.764, 'lng': -86.156},
    'LAC': {'altitude': 340, 'tz': 'PT', 'arena': 'Intuit Dome', 'lat': 33.944, 'lng': -118.341},
    'LAL': {'altitude': 340, 'tz': 'PT', 'arena': 'Crypto.com Arena', 'lat': 34.043, 'lng': -118.267},
    'MEM': {'altitude': 337, 'tz': 'CT', 'arena': 'FedExForum', 'lat': 35.138, 'lng': -90.051},
    'MIA': {'altitude': 6, 'tz': 'ET', 'arena': 'Kaseya Center', 'lat': 25.781, 'lng': -80.187},
    'MIL': {'altitude': 617, 'tz': 'CT', 'arena': 'Fiserv Forum', 'lat': 43.045, 'lng': -87.917},
    'MIN': {'altitude': 830, 'tz': 'CT', 'arena': 'Target Center', 'lat': 44.980, 'lng': -93.276},
    'NOP': {'altitude': 3, 'tz': 'CT', 'arena': 'Smoothie King Center', 'lat': 29.949, 'lng': -90.082},
    'NYK': {'altitude': 33, 'tz': 'ET', 'arena': 'Madison Square Garden', 'lat': 40.751, 'lng': -73.994},
    'OKC': {'altitude': 1201, 'tz': 'CT', 'arena': 'Paycom Center', 'lat': 35.463, 'lng': -97.515},
    'ORL': {'altitude': 82, 'tz': 'ET', 'arena': 'Amway Center', 'lat': 28.539, 'lng': -81.384},
    'PHI': {'altitude': 39, 'tz': 'ET', 'arena': 'Wells Fargo Center', 'lat': 39.901, 'lng': -75.172},
    'PHX': {'altitude': 1086, 'tz': 'MT', 'arena': 'Footprint Center', 'lat': 33.446, 'lng': -112.071},
    'POR': {'altitude': 50, 'tz': 'PT', 'arena': 'Moda Center', 'lat': 45.532, 'lng': -122.667},
    'SAC': {'altitude': 30, 'tz': 'PT', 'arena': 'Golden 1 Center', 'lat': 38.580, 'lng': -121.500},
    'SAS': {'altitude': 650, 'tz': 'CT', 'arena': 'Frost Bank Center', 'lat': 29.427, 'lng': -98.438},
    'TOR': {'altitude': 249, 'tz': 'ET', 'arena': 'Scotiabank Arena', 'lat': 43.643, 'lng': -79.379},
    'UTA': {'altitude': 4226, 'tz': 'MT', 'arena': 'Delta Center', 'lat': 40.768, 'lng': -111.901},
    'WAS': {'altitude': 0, 'tz': 'ET', 'arena': 'Capital One Arena', 'lat': 38.898, 'lng': -77.021},
}

# Timezone ordinal for computing cross-timezone travel
TZ_ORDINAL = {'ET': 0, 'CT': 1, 'MT': 2, 'PT': 3}

# Team abbreviation to short name mapping (reverse of nba_fetcher.TEAM_ABR)
ABR_TO_SHORT = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}

SHORT_TO_ABR = {v: k for k, v in ABR_TO_SHORT.items()}


def get_venue_altitude(home_team_abr):
    """Get altitude in feet for the home team's arena."""
    venue = VENUE_MAP.get(home_team_abr)
    return venue['altitude'] if venue else 0


def get_travel_zone_diff(away_team_abr, home_team_abr):
    """Get timezone difference (0-3) between away and home teams."""
    away_venue = VENUE_MAP.get(away_team_abr)
    home_venue = VENUE_MAP.get(home_team_abr)
    if not away_venue or not home_venue:
        return 0
    away_tz = TZ_ORDINAL.get(away_venue['tz'], 0)
    home_tz = TZ_ORDINAL.get(home_venue['tz'], 0)
    return abs(away_tz - home_tz)


import math

def haversine_miles(lat1, lng1, lat2, lng2):
    """Calculate great-circle distance in miles between two points."""
    R = 3959  # Earth radius in miles
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_travel_distance(from_abr, to_abr):
    """Get distance in miles between two NBA arenas by team abbreviation."""
    from_venue = VENUE_MAP.get(from_abr)
    to_venue = VENUE_MAP.get(to_abr)
    if not from_venue or not to_venue:
        return 0
    return round(haversine_miles(
        from_venue['lat'], from_venue['lng'],
        to_venue['lat'], to_venue['lng']
    ))
