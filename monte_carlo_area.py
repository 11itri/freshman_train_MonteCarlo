import argparse
import json
import random
from pathlib import Path

from shapely.geometry import shape, Point
from shapely.ops import unary_union, transform as shapely_transform
from pyproj import CRS, Transformer
from typing import Optional


def load_union_geometry(geojson_path: Path):
    """GeoJSONファイルから全フィーチャのジオメトリを読み込み、Unionしたものを返す。"""
    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    geoms = []
    for feature in data.get("features", []):
        geom_dict = feature.get("geometry")
        if not geom_dict:
            continue
        geoms.append(shape(geom_dict))

    if not geoms:
        raise ValueError("GeoJSONにジオメトリが含まれていません")

    return unary_union(geoms)


def build_equal_area_transformer():
    """JGD2011(EPSG:6668) -> 日本付近用の正積投影(AEA)への変換器を作成する。

    千葉県付近(東経140度・北緯35度)に合わせたAlbers Equal-Areaを使い、
    変換後座標系の単位はメートルになる。
    """
    # 入力: JGD2011 geographic 2D
    src = CRS.from_epsg(6668)

    # 出力: 日本周辺に適したAlbers Equal-Area (任意定義)
    # 標準緯線(lat_1, lat_2)を 30度・40度、中心(lat_0, lon_0)を 35度・140度に設定
    dst = CRS.from_proj4(
        "+proj=aea +lat_1=30 +lat_2=40 +lat_0=35 "
        "+lon_0=140 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return transformer


def project_geometry(geom, transformer: Transformer):
    """Shapelyジオメトリを与えられた変換器で投影する。"""
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def monte_carlo_area(geom, num_samples: int, rng_seed: Optional[int] = None):
    """Monte Carlo法でポリゴンの面積を推定する。

    geom: ShapelyのPolygon/MultiPolygon（既に正積投影座標系上にある前提）
    num_samples: 乱数サンプル数
    rng_seed: 再現性確保のためのシード
    戻り値: (推定面積[m^2], bbox面積[m^2], ヒット数)
    """
    if num_samples <= 0:
        raise ValueError("num_samples は正の整数である必要があります")

    if rng_seed is not None:
        random.seed(rng_seed)

    minx, miny, maxx, maxy = geom.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= 0:
        raise ValueError("ジオメトリのバウンディングボックス面積が0以下です")

    inside_count = 0
    for _ in range(num_samples):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        # contains() だと境界上を除外するので、covers() を使う
        if geom.covers(p):
            inside_count += 1

    area_est = bbox_area * inside_count / num_samples
    return area_est, bbox_area, inside_count


def parse_args():
    parser = argparse.ArgumentParser(description="Monte Carlo法で千葉県の面積を推定するスクリプト")
    parser.add_argument(
        "geojson",
        type=Path,
        help="N03のGeoJSONファイルパス（例: N03-23_12_230101.geojson）",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=200_000,
        help="Monte Carloサンプル数 (デフォルト: 200000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="乱数シード（再現性確保用、デフォルト: 0）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    geojson_path: Path = args.geojson
    if not geojson_path.exists():
        raise SystemExit(f"GeoJSONファイルが見つかりません: {geojson_path}")

    print(f"GeoJSON読み込み中: {geojson_path}")
    union_geom_ll = load_union_geometry(geojson_path)

    # 投影
    print("座標変換(正積投影)中...")
    transformer = build_equal_area_transformer()
    union_geom_proj = project_geometry(union_geom_ll, transformer)

    # 解析的(Shapely)面積も計算して比較できるようにしておく
    analytic_area = union_geom_proj.area  # [m^2]

    print(f"Monte Carloサンプル数: {args.num_samples}")
    est_area, bbox_area, inside = monte_carlo_area(
        union_geom_proj, num_samples=args.num_samples, rng_seed=args.seed
    )

    print("--- 結果 ---")
    print(f"バウンディングボックス面積: {bbox_area:,.0f} m^2")
    print(f"内側に含まれた数: {inside}")
    print(f"外側になった数: {args.num_samples - inside}")
    print(f"Monte Carlo推定面積: {est_area:,.0f} m^2 ({est_area / 1e6:,.2f} km^2)")
    print(f"Shapelyによる面積(参考): {analytic_area:,.0f} m^2 ({analytic_area / 1e6:,.2f} km^2)")
    if analytic_area > 0:
        rel_err = abs(est_area - analytic_area) / analytic_area
        print(f"Monte Carlo推定の相対誤差(参考): {rel_err * 100:.3f} %")


if __name__ == "__main__":
    main()
