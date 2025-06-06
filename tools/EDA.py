import re
import matplotlib.pyplot as plt
import pandas as pd

def plot_vehicles_time_pdf(log_text, pdf_filename="vehicles_vs_time.pdf"):
    # Extract time and count from each log line
    pattern = r'(\d{2}:\d{2}:\d{2}:\d{2}:\d{2}) vehicles in route: (\d+)'
    times = ['00:00']
    counts = [0]
    for match in re.finditer(pattern, log_text):
        # Reformat as HH:MM for display, using hour+minute in the correct order
        time_str = match.group(1)
        # Parse as: DD:HH:MM:SS:ms (from your format)
        ww, dd, hh, mm, ss = time_str.split(":")
        display_time = f"{hh}:{mm}"
        times.append(display_time)
        counts.append(int(match.group(2)))
    # Make DataFrame
    df = pd.DataFrame({'time': times, 'vehicles': counts})
    # Plot
    plt.figure(figsize=(16,6))
    plt.plot(df['vehicles'], marker='.', linewidth=1.5)
    plt.title('Vehicles in Route vs. Time', fontsize=18)
    plt.ylabel('Vehicles in Route', fontsize=15)
    plt.xlabel('Time of Day', fontsize=15)
    # Label every hour for X axis
    step_per_hour = 12
    plt.xticks(
        range(0, len(df), step_per_hour),
        [df['time'][i] for i in range(0, len(df), step_per_hour)],
        rotation=45,
        fontsize=12
    )
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    # Save as PDF (vector, best for publication)
    plt.savefig(pdf_filename)
    plt.show()
    plt.close()

# Usage: paste your log in place of 'log_text'

full_log = """
00:00:00:05:00 vehicles in route: 15
00:00:00:10:00 vehicles in route: 28
00:00:00:15:00 vehicles in route: 33
00:00:00:20:00 vehicles in route: 30
00:00:00:25:00 vehicles in route: 33
00:00:00:30:00 vehicles in route: 36
00:00:00:35:00 vehicles in route: 34
00:00:00:40:00 vehicles in route: 35
00:00:00:45:00 vehicles in route: 28
00:00:00:50:00 vehicles in route: 29
00:00:00:55:00 vehicles in route: 35
00:00:01:00:00 vehicles in route: 28
00:00:01:05:00 vehicles in route: 32
00:00:01:10:00 vehicles in route: 34
00:00:01:15:00 vehicles in route: 31
00:00:01:20:00 vehicles in route: 33
00:00:01:25:00 vehicles in route: 34
00:00:01:30:00 vehicles in route: 38
00:00:01:35:00 vehicles in route: 32
00:00:01:40:00 vehicles in route: 35
00:00:01:45:00 vehicles in route: 33
00:00:01:50:00 vehicles in route: 34
00:00:01:55:00 vehicles in route: 35
00:00:02:00:00 vehicles in route: 37
00:00:02:05:00 vehicles in route: 32
00:00:02:10:00 vehicles in route: 31
00:00:02:15:00 vehicles in route: 26
00:00:02:20:00 vehicles in route: 27
00:00:02:25:00 vehicles in route: 31
00:00:02:30:00 vehicles in route: 30
00:00:02:35:00 vehicles in route: 31
00:00:02:40:00 vehicles in route: 32
00:00:02:45:00 vehicles in route: 31
00:00:02:50:00 vehicles in route: 29
00:00:02:55:00 vehicles in route: 32
00:00:03:00:00 vehicles in route: 35
00:00:03:05:00 vehicles in route: 35
00:00:03:10:00 vehicles in route: 33
00:00:03:15:00 vehicles in route: 33
00:00:03:20:00 vehicles in route: 33
00:00:03:25:00 vehicles in route: 31
00:00:03:30:00 vehicles in route: 30
00:00:03:35:00 vehicles in route: 34
00:00:03:40:00 vehicles in route: 31
00:00:03:45:00 vehicles in route: 32
00:00:03:50:00 vehicles in route: 31
00:00:03:55:00 vehicles in route: 36
00:00:04:00:00 vehicles in route: 33
00:00:04:05:00 vehicles in route: 30
00:00:04:10:00 vehicles in route: 30
00:00:04:15:00 vehicles in route: 30
00:00:04:20:00 vehicles in route: 31
00:00:04:25:00 vehicles in route: 31
00:00:04:30:00 vehicles in route: 31
00:00:04:35:00 vehicles in route: 28
00:00:04:40:00 vehicles in route: 30
00:00:04:45:00 vehicles in route: 29
00:00:04:50:00 vehicles in route: 36
00:00:04:55:00 vehicles in route: 32
00:00:05:00:00 vehicles in route: 30
00:00:05:05:00 vehicles in route: 141
00:00:05:10:00 vehicles in route: 206
00:00:05:15:00 vehicles in route: 261
00:00:05:20:00 vehicles in route: 252
00:00:05:25:00 vehicles in route: 266
00:00:05:30:00 vehicles in route: 260
00:00:05:35:00 vehicles in route: 263
00:00:05:40:00 vehicles in route: 245
00:00:05:45:00 vehicles in route: 249
00:00:05:50:00 vehicles in route: 259
00:00:05:55:00 vehicles in route: 272
00:00:06:00:00 vehicles in route: 271
00:00:06:05:00 vehicles in route: 272
00:00:06:10:00 vehicles in route: 266
00:00:06:15:00 vehicles in route: 251
00:00:06:20:00 vehicles in route: 254
00:00:06:25:00 vehicles in route: 267
00:00:06:30:00 vehicles in route: 252
00:00:06:35:00 vehicles in route: 357
00:00:06:40:00 vehicles in route: 471
00:00:06:45:00 vehicles in route: 547
00:00:06:50:00 vehicles in route: 577
00:00:06:55:00 vehicles in route: 582
00:00:07:00:00 vehicles in route: 614
00:00:07:05:00 vehicles in route: 624
00:00:07:10:00 vehicles in route: 642
00:00:07:15:00 vehicles in route: 670
00:00:07:20:00 vehicles in route: 673
00:00:07:25:00 vehicles in route: 689
00:00:07:30:00 vehicles in route: 713
00:00:07:35:00 vehicles in route: 707
00:00:07:40:00 vehicles in route: 731
00:00:07:45:00 vehicles in route: 743
00:00:07:50:00 vehicles in route: 767
00:00:07:55:00 vehicles in route: 743
00:00:08:00:00 vehicles in route: 760
00:00:08:05:00 vehicles in route: 780
00:00:08:10:00 vehicles in route: 784
00:00:08:15:00 vehicles in route: 795
00:00:08:20:00 vehicles in route: 809
00:00:08:25:00 vehicles in route: 808
00:00:08:30:00 vehicles in route: 832
00:00:08:35:00 vehicles in route: 818
00:00:08:40:00 vehicles in route: 835
00:00:08:45:00 vehicles in route: 863
00:00:08:50:00 vehicles in route: 868
00:00:08:55:00 vehicles in route: 883
00:00:09:00:00 vehicles in route: 880
00:00:09:05:00 vehicles in route: 896
00:00:09:10:00 vehicles in route: 934
00:00:09:15:00 vehicles in route: 941
00:00:09:20:00 vehicles in route: 967
00:00:09:25:00 vehicles in route: 948
00:00:09:30:00 vehicles in route: 945
00:00:09:35:00 vehicles in route: 906
00:00:09:40:00 vehicles in route: 813
00:00:09:45:00 vehicles in route: 764
00:00:09:50:00 vehicles in route: 703
00:00:09:55:00 vehicles in route: 694
00:00:10:00:00 vehicles in route: 662
00:00:10:05:00 vehicles in route: 632
00:00:10:10:00 vehicles in route: 592
00:00:10:15:00 vehicles in route: 558
00:00:10:20:00 vehicles in route: 562
00:00:10:25:00 vehicles in route: 539
00:00:10:30:00 vehicles in route: 541
00:00:10:35:00 vehicles in route: 539
00:00:10:40:00 vehicles in route: 547
00:00:10:45:00 vehicles in route: 549
00:00:10:50:00 vehicles in route: 535
00:00:10:55:00 vehicles in route: 531
00:00:11:00:00 vehicles in route: 531
00:00:11:05:00 vehicles in route: 530
00:00:11:10:00 vehicles in route: 523
00:00:11:15:00 vehicles in route: 525
00:00:11:20:00 vehicles in route: 516
00:00:11:25:00 vehicles in route: 519
00:00:11:30:00 vehicles in route: 500
00:00:11:35:00 vehicles in route: 490
00:00:11:40:00 vehicles in route: 485
00:00:11:45:00 vehicles in route: 486
00:00:11:50:00 vehicles in route: 496
00:00:11:55:00 vehicles in route: 479
00:00:12:00:00 vehicles in route: 466
00:00:12:05:00 vehicles in route: 422
00:00:12:10:00 vehicles in route: 376
00:00:12:15:00 vehicles in route: 367
00:00:12:20:00 vehicles in route: 337
00:00:12:25:00 vehicles in route: 309
00:00:12:30:00 vehicles in route: 287
00:00:12:35:00 vehicles in route: 298
00:00:12:40:00 vehicles in route: 290
00:00:12:45:00 vehicles in route: 262
00:00:12:50:00 vehicles in route: 262
00:00:12:55:00 vehicles in route: 263
00:00:13:00:00 vehicles in route: 251
00:00:13:05:00 vehicles in route: 257
00:00:13:10:00 vehicles in route: 265
00:00:13:15:00 vehicles in route: 268
00:00:13:20:00 vehicles in route: 258
00:00:13:25:00 vehicles in route: 256
00:00:13:30:00 vehicles in route: 248
00:00:13:35:00 vehicles in route: 308
00:00:13:40:00 vehicles in route: 315
00:00:13:45:00 vehicles in route: 323
00:00:13:50:00 vehicles in route: 319
00:00:13:55:00 vehicles in route: 326
00:00:14:00:00 vehicles in route: 334
00:00:14:05:00 vehicles in route: 346
00:00:14:10:00 vehicles in route: 347
00:00:14:15:00 vehicles in route: 353
00:00:14:20:00 vehicles in route: 336
00:00:14:25:00 vehicles in route: 341
00:00:14:30:00 vehicles in route: 342
00:00:14:35:00 vehicles in route: 350
00:00:14:40:00 vehicles in route: 332
00:00:14:45:00 vehicles in route: 343
00:00:14:50:00 vehicles in route: 354
00:00:14:55:00 vehicles in route: 354
00:00:15:00:00 vehicles in route: 338
00:00:15:05:00 vehicles in route: 330
00:00:15:10:00 vehicles in route: 346
00:00:15:15:00 vehicles in route: 354
00:00:15:20:00 vehicles in route: 343
00:00:15:25:00 vehicles in route: 344
00:00:15:30:00 vehicles in route: 350
00:00:15:35:00 vehicles in route: 358
00:00:15:40:00 vehicles in route: 375
00:00:15:45:00 vehicles in route: 349
00:00:15:50:00 vehicles in route: 360
00:00:15:55:00 vehicles in route: 356
00:00:16:00:00 vehicles in route: 345
00:00:16:05:00 vehicles in route: 226
00:00:16:10:00 vehicles in route: 96
00:00:16:15:00 vehicles in route: 36
00:00:16:20:00 vehicles in route: 33
00:00:16:25:00 vehicles in route: 32
00:00:16:30:00 vehicles in route: 29
00:00:16:35:00 vehicles in route: 33
00:00:16:40:00 vehicles in route: 33
00:00:16:45:00 vehicles in route: 32
00:00:16:50:00 vehicles in route: 32
00:00:16:55:00 vehicles in route: 35
00:00:17:00:00 vehicles in route: 30
00:00:17:05:00 vehicles in route: 34
00:00:17:10:00 vehicles in route: 34
00:00:17:15:00 vehicles in route: 30
00:00:17:20:00 vehicles in route: 30
00:00:17:25:00 vehicles in route: 33
00:00:17:30:00 vehicles in route: 33
00:00:17:35:00 vehicles in route: 35
00:00:17:40:00 vehicles in route: 35
00:00:17:45:00 vehicles in route: 37
00:00:17:50:00 vehicles in route: 30
00:00:17:55:00 vehicles in route: 33
00:00:18:00:00 vehicles in route: 29
00:00:18:05:00 vehicles in route: 34
00:00:18:10:00 vehicles in route: 37
00:00:18:15:00 vehicles in route: 38
00:00:18:20:00 vehicles in route: 38
00:00:18:25:00 vehicles in route: 30
00:00:18:30:00 vehicles in route: 31
00:00:18:35:00 vehicles in route: 31
00:00:18:40:00 vehicles in route: 33
00:00:18:45:00 vehicles in route: 33
00:00:18:50:00 vehicles in route: 34
00:00:18:55:00 vehicles in route: 32
00:00:19:00:00 vehicles in route: 30
00:00:19:05:00 vehicles in route: 83
00:00:19:10:00 vehicles in route: 115
00:00:19:15:00 vehicles in route: 129
00:00:19:20:00 vehicles in route: 137
00:00:19:25:00 vehicles in route: 145
00:00:19:30:00 vehicles in route: 155
00:00:19:35:00 vehicles in route: 150
00:00:19:40:00 vehicles in route: 153
00:00:19:45:00 vehicles in route: 160
00:00:19:50:00 vehicles in route: 139
00:00:19:55:00 vehicles in route: 129
00:00:20:00:00 vehicles in route: 145
00:00:20:05:00 vehicles in route: 142
00:00:20:10:00 vehicles in route: 137
00:00:20:15:00 vehicles in route: 147
00:00:20:20:00 vehicles in route: 142
00:00:20:25:00 vehicles in route: 143
00:00:20:30:00 vehicles in route: 131
00:00:20:35:00 vehicles in route: 134
00:00:20:40:00 vehicles in route: 147
00:00:20:45:00 vehicles in route: 133
00:00:20:50:00 vehicles in route: 137
00:00:20:55:00 vehicles in route: 149
00:00:21:00:00 vehicles in route: 150
00:00:21:05:00 vehicles in route: 140
00:00:21:10:00 vehicles in route: 144
00:00:21:15:00 vehicles in route: 137
00:00:21:20:00 vehicles in route: 138
00:00:21:25:00 vehicles in route: 139
00:00:21:30:00 vehicles in route: 146
00:00:21:35:00 vehicles in route: 136
00:00:21:40:00 vehicles in route: 141
00:00:21:45:00 vehicles in route: 150
00:00:21:50:00 vehicles in route: 158
00:00:21:55:00 vehicles in route: 156
00:00:22:00:00 vehicles in route: 160
00:00:22:05:00 vehicles in route: 142
00:00:22:10:00 vehicles in route: 147
00:00:22:15:00 vehicles in route: 138
00:00:22:20:00 vehicles in route: 145
00:00:22:25:00 vehicles in route: 145
00:00:22:30:00 vehicles in route: 129
00:00:22:35:00 vehicles in route: 145
00:00:22:40:00 vehicles in route: 144
00:00:22:45:00 vehicles in route: 150
00:00:22:50:00 vehicles in route: 148
00:00:22:55:00 vehicles in route: 144
00:00:23:00:00 vehicles in route: 138
00:00:23:05:00 vehicles in route: 98
00:00:23:10:00 vehicles in route: 63
00:00:23:15:00 vehicles in route: 40
00:00:23:20:00 vehicles in route: 33
00:00:23:25:00 vehicles in route: 33
00:00:23:30:00 vehicles in route: 33
00:00:23:35:00 vehicles in route: 35
00:00:23:40:00 vehicles in route: 32
00:00:23:45:00 vehicles in route: 33
00:00:23:50:00 vehicles in route: 29
00:00:23:55:00 vehicles in route: 33
00:01:00:00:00 vehicles in route: 29
"""
plot_vehicles_time_pdf(full_log)
