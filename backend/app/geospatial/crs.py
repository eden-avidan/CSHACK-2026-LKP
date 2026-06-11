import numpy as np
from pyproj import CRS, Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


class CRSHelper:
    """LKP-anchored WGS84 ↔ UTM transforms."""

    def __init__(self, lat: float, lon: float) -> None:
        self.lat = lat
        self.lon = lon
        self.epsg = utm_epsg(lon, lat)
        self._to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._to_wgs = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        self.origin_e, self.origin_n = self._to_utm.transform(lon, lat)

    def to_utm(self, lon: float, lat: float) -> tuple[float, float]:
        e, n = self._to_utm.transform(lon, lat)
        return float(e), float(n)

    def to_utm_array(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch WGS84 -> UTM transform for arrays of lon/lat."""
        e, n = self._to_utm.transform(np.asarray(lons), np.asarray(lats))
        return np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64)

    def to_wgs84(self, easting: float, northing: float) -> tuple[float, float]:
        lon, lat = self._to_wgs.transform(easting, northing)
        return float(lat), float(lon)

    def offset_to_wgs84(self, de: float, dn: float) -> tuple[float, float]:
        return self.to_wgs84(self.origin_e + de, self.origin_n + dn)
