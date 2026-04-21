# Advanced Level-Wise Hyperparameter Sweep Summary

- Base config: `configs/curriculum/experiment6.yaml`
- Episodes per level: `2000`
- Seeds: `42, 43, 44, 45, 46`
- Optimal success: reach the goal in the shortest number of steps while collecting all bonus cells on bonus levels.

## level1

### Alpha

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.000 | 0.200 | 12.98 | 35.00 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.15 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.3 | 1.000 | 0.400 | 13.22 | 33.80 | 1.000 | 1.000 | 0.000 | 1.000 |

### Gamma

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 1.000 | 0.600 | 13.44 | 33.80 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.95 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.99 | 1.000 | 0.600 | 13.22 | 33.80 | 1.000 | 1.000 | 0.000 | 1.000 |

### Epsilon Decay

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.995 | 1.000 | 0.200 | 12.99 | 34.40 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.997 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.999 | 1.000 | 1.000 | 13.60 | 33.00 | 1.000 | 1.000 | 0.000 | 1.000 |

### Epsilon Min

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.05 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |
| 0.1 | 1.000 | 0.800 | 13.45 | 33.20 | 1.000 | 1.000 | 0.000 | 1.000 |

## level2

### Alpha

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.000 | 1.000 | 17.20 | 15.00 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.15 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.3 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |

### Gamma

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 1.000 | 0.600 | 16.90 | 15.40 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.95 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.99 | 1.000 | 0.600 | 16.90 | 15.40 | 0.000 | 0.000 | 0.000 | 1.000 |

### Epsilon Decay

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.995 | 1.000 | 0.400 | 16.75 | 15.60 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.997 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.999 | 1.000 | 0.800 | 17.05 | 15.20 | 0.000 | 0.000 | 0.000 | 1.000 |

### Epsilon Min

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.05 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.1 | 1.000 | 0.200 | 16.60 | 15.80 | 0.000 | 0.000 | 0.000 | 1.000 |

## level3

### Alpha

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.000 | 0.000 | 18.82 | 21.80 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.15 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.3 | 1.000 | 0.000 | 18.17 | 19.60 | 0.000 | 0.000 | 1.000 | 0.000 |

### Gamma

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 1.000 | 0.000 | 18.25 | 19.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.95 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.99 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |

### Epsilon Decay

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.995 | 1.000 | 0.000 | 17.87 | 20.00 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.997 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.999 | 1.000 | 0.000 | 18.40 | 19.00 | 0.000 | 0.000 | 1.000 | 0.000 |

### Epsilon Min

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.05 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.1 | 1.000 | 0.000 | 18.56 | 20.20 | 0.000 | 0.000 | 1.000 | 0.000 |

## level4

### Alpha

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.000 | 0.000 | 13.95 | 39.60 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.15 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.3 | 1.000 | 0.000 | 13.80 | 39.80 | 1.000 | 1.000 | 1.000 | 0.000 |

### Gamma

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 1.000 | 0.000 | 14.10 | 39.40 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.95 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.99 | 1.000 | 0.000 | 13.95 | 39.60 | 1.000 | 1.000 | 1.000 | 0.000 |

### Epsilon Decay

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.995 | 1.000 | 0.000 | 13.95 | 39.60 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.997 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.999 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |

### Epsilon Min

| Value | Final Success | Final Optimal Success | Avg Reward | Avg Steps (Success) | Key Rate | Door Rate | Any Bonus Rate | All Bonus Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.05 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
| 0.1 | 1.000 | 0.000 | 14.25 | 39.20 | 1.000 | 1.000 | 1.000 | 0.000 |
