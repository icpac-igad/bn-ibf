| Colour                                                                           | HTML code | Level                       | Classification description                                                                                                    |
| -------------------------------------------------------------------------------- | --------: | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| <span style="background:#FFFF00;padding:2px 20px;border:1px solid #999;"></span> | `#FFFF00` | Watch                       | A relevant precipitation deficit is observed                                                                                  |
| <span style="background:#FFA600;padding:2px 20px;border:1px solid #999;"></span> | `#FFA600` | Warning                     | The above precipitation deficit is accompanied by soil moisture deficit                                                       |
| <span style="background:#FE0000;padding:2px 20px;border:1px solid #999;"></span> | `#FE0000` | Alert                       | The above two conditions are accompanied by a negative anomaly of vegetation growth                                           |
| <span style="background:#9F8001;padding:2px 20px;border:1px solid #999;"></span> | `#9F8001` | Partial recovery            | After a drought episode, the meteorological conditions have recovered to normal, but vegetation conditions are yet to recover |
| <span style="background:#9EC75F;padding:2px 20px;border:1px solid #999;"></span> | `#9EC75F` | Full recovery of vegetation | After a drought episode, both the meteorological and vegetation conditions have recovered to normal                           |
| <span style="background:#F5F5F5;padding:2px 20px;border:1px solid #999;"></span> | `#F5F5F5` | No drought conditions       | No drought conditions                                                                                                         |


## Table for cacualting CDI 

| CDI | Level            | SPI9/12 < -1 | SPI3 < -1 | SPI1 < -2 | SPI3_prev < -1 | SPI1_prev < -2 | SMA < -1 | fAPAR < -1 |
| --- | ---------------- | ------------ | --------- | --------- | -------------- | -------------- | -------- | ---------- |
| 1   | Watch            |              |           | X         |                |                |          |            |
| 2   | Watch            |              | X         |           |                |                |          |            |
| 3   | Watch            | X            | X         |           |                |                |          |            |
| 4   | Warning          |              |           | X         |                |                | X        |            |
| 5   | Warning          |              | X         |           |                |                | X        |            |
| 6   | Warning          | X            | X         |           |                |                | X        |            |
| 7   | Alert            |              |           | X         |                |                |          | X          |
| 8   | Alert            |              | X         |           |                |                |          | X          |
| 9   | Alert            |              | X         |           |                |                | X        | X          |
| 10  | Alert            | X            | X         |           |                |                | X        | X          |
| 11  | Partial recovery |              |           |           |                | X              |          | X          |
| 12  | Partial recovery |              |           |           | X              |                |          | X          |
| 13  | Full recovery    |              |           |           |                | X              |          |            |
| 14  | Full recovery    |              |           |           | X              |                |          |            |


## script 

def calculate_cdi(
    spi9_12_lt_m1=False,
    spi3_lt_m1=False,
    spi1_lt_m2=False,
    spi3_prev_lt_m1=False,
    spi1_prev_lt_m2=False,
    sma_lt_m1=False,
    fapar_lt_m1=False,
):
    """
    Calculate CDI class using the reconstructed convergence-of-evidence table.

    Inputs are Boolean indicators:
    True  = condition is met
    False = condition is not met
    """

    # Alert: precipitation shortage + vegetation anomaly
    if fapar_lt_m1 and sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 10, "Alert"

    if fapar_lt_m1 and sma_lt_m1 and spi3_lt_m1:
        return 9, "Alert"

    if fapar_lt_m1 and spi3_lt_m1:
        return 8, "Alert"

    if fapar_lt_m1 and spi1_lt_m2:
        return 7, "Alert"

    # Warning: precipitation shortage + soil moisture anomaly
    if sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 6, "Warning"

    if sma_lt_m1 and spi3_lt_m1:
        return 5, "Warning"

    if sma_lt_m1 and spi1_lt_m2:
        return 4, "Warning"

    # Watch: precipitation shortage only
    if spi9_12_lt_m1 and spi3_lt_m1:
        return 3, "Watch"

    if spi3_lt_m1:
        return 2, "Watch"

    if spi1_lt_m2:
        return 1, "Watch"

    # Partial recovery: previous precipitation deficit + vegetation anomaly
    if fapar_lt_m1 and spi3_prev_lt_m1:
        return 12, "Partial recovery"

    if fapar_lt_m1 and spi1_prev_lt_m2:
        return 11, "Partial recovery"

    # Full recovery: previous precipitation deficit only
    if spi3_prev_lt_m1:
        return 14, "Full recovery of vegetation"

    if spi1_prev_lt_m2:
        return 13, "Full recovery of vegetation"

    return 0, "No drought conditions"