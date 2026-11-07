import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker as ticker  # 确保在文件顶部或函数内导入
import matplotlib.pyplot as plt

# Keep fonts editable in PDF files.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
ratio_labels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

datasets = [
    'ALOI', 'Caltech7', 'ESP-Game', 'IAPR-TC12', 'NUSWIDE20K', 'OutdoorScene',
]

method_order = [
    'DUANet', 'MMdynamics', 'LGCNFF', 'PDMF', 'RCML', 'KAMSSM', 'CurvFlow', 'GTMC-HOA', 'MDGG', 'MvDHP',
]

metric_labels = {
    "ACC": "ACC",
    "F1": "F1-score",
}

data = {
    'ACC': {
        'DUANet': {
            'ALOI': [58.2119, 92.1156, 95.3645, 96.7907, 97.4536, 97.0487, 97.4571, 97.8120, 98.2034, 98.4045],
            'Caltech7': [80.9006, 86.0483, 88.8428, 89.5674, 89.9186, 90.6680, 90.9072, 91.4480, 91.4920, 91.8207],
            'NUSWIDE20K': [28.4421, 30.6656, 32.2835, 33.5962, 34.5947, 35.4886, 36.1400, 36.7267, 37.2073, 37.4880],
            'OutdoorScene': [55.6147, 59.4047, 62.3020, 63.9070, 64.6726, 65.9830, 66.5063, 66.4228, 67.7890, 68.9881],
            'ESP-Game': [44.5978, 47.9424, 49.0135, 49.1706, 50.1958, 50.5905, 51.2022, 51.6591, 51.8444, 52.5644],
            'IAPR-TC12': [35.6479, 37.2132, 39.4248, 41.2572, 41.8632, 41.3457, 42.4481, 43.2160, 43.2030, 43.2587],
        },
        'MMdynamics': {
            'ALOI': [45.0146, 51.5996, 53.6670, 35.8837, 59.9506, 60.3426, 51.9143, 60.6163, 55.6610, 60.7792],
            'Caltech7': [84.0600, 85.0830, 85.6504, 85.6997, 85.7919, 86.0019, 86.1522, 85.9729, 85.9926, 86.0598],
            'NUSWIDE20K': [61.3095, 63.2100, 64.4271, 65.0112, 65.3573, 65.7557, 66.0585, 66.0433, 66.3455, 66.4540],
            'OutdoorScene': [58.2537, 61.2484, 65.6455, 67.5256, 68.7103, 69.3092, 69.8053, 70.3906, 70.6964, 70.9524],
            'ESP-Game': [71.2129, 72.6518, 74.1602, 74.9581, 75.6286, 76.0114, 76.2064, 76.3765, 76.5551, 76.5615],
            'IAPR-TC12': [50.1916, 53.3682, 55.0899, 56.0694, 56.6842, 57.5523, 57.7007, 58.0187, 58.4541, 58.7016],
        },
        'LGCNFF': {
            'ALOI': [90.5360, 95.4000, 96.2580, 96.9080, 97.4760, 97.4960, 97.3160, 97.4100, 97.1540, 97.4760],
            'Caltech7': [87.2900, 90.7860, 92.1800, 92.7060, 92.7420, 92.3540, 92.9080, 93.0320, 93.0200, 93.4520],
            'NUSWIDE20K': [24.4640, 26.0620, 26.6960, 28.9900, 29.4260, 29.4000, 26.7700, 27.0860, 25.3640, 27.1180],
            'OutdoorScene': [71.5740, 75.8920, 77.1820, 78.2520, 78.5420, 79.0740, 79.1300, 79.4680, 79.4600, 79.5080],
            'ESP-Game': [62.5420, 69.2640, 67.5300, 68.1780, 70.0360, 70.6080, 71.1020, 71.6060, 67.9780, 70.9760],
            'IAPR-TC12': [50.5480, 55.8800, 58.0140, 59.5200, 60.3120, 60.5820, 61.6460, 61.9420, 62.1900, 62.7580],
        },
        'PDMF': {
            'ALOI': [23.4402, 20.3509, 26.4418, 22.1163, 29.4685, 38.9460, 26.9714, 40.4314, 33.5254, 39.2950],
            'Caltech7': [57.4982, 77.0437, 76.3927, 76.0305, 70.2443, 78.3156, 69.0928, 72.1041, 68.8779, 71.7935],
            'NUSWIDE20K': [31.6411, 56.2489, 56.8047, 58.8200, 65.3440, 72.9486, 75.4892, 73.8550, 75.9673, 76.2580],
            'OutdoorScene': [64.6437, 59.7354, 63.1860, 57.8977, 56.8849, 53.9851, 51.5349, 68.8035, 72.0352, 71.8452],
            'ESP-Game': [41.7597, 68.0935, 60.0661, 64.2601, 64.7389, 62.7817, 66.9428, 72.4660, 74.2476, 81.3275],
            'IAPR-TC12': [47.0320, 48.0719, 57.2229, 60.4137, 59.3959, 58.6834, 59.5926, 53.5469, 61.8005, 61.7974],
        },
        'RCML': {
            'ALOI': [52.7114, 82.7451, 82.9597, 94.9302, 94.2645, 94.3083, 93.8000, 97.8120, 97.7966, 97.6252],
            'Caltech7': [91.6655, 91.5385, 92.6257, 93.6217, 93.9367, 93.2236, 96.4338, 96.8552, 97.2380, 97.4185],
            'NUSWIDE20K': [67.2926, 71.6611, 72.8165, 73.0000, 73.6573, 74.1614, 74.4615, 74.8067, 74.7691, 74.8920],
            'OutdoorScene': [70.0078, 74.0471, 75.8512, 78.3907, 79.1071, 78.3528, 80.0573, 80.1612, 80.0541, 82.0982],
            'ESP-Game': [80.9199, 83.0577, 84.7457, 84.1854, 86.4192, 86.9205, 87.8884, 87.9964, 88.1952, 88.9699],
            'IAPR-TC12': [54.9725, 60.1443, 64.8862, 67.0624, 67.9043, 69.1144, 69.8825, 69.7921, 69.4654, 70.8758],
        },
        'KAMSSM': {
            'ALOI': [87.8500, 94.7000, 94.9300, 95.0200, 93.2000, 95.8100, 85.2600, 96.8600, 96.2700, 97.4400],
            'Caltech7': [88.3600, 94.5100, 94.8800, 95.4400, 96.0500, 96.8200, 97.0000, 97.2400, 97.1600, 97.2000],
            'NUSWIDE20K': [65.1600, 65.2900, 69.4400, 72.4400, 74.2500, 75.1900, 75.7800, 76.2200, 76.2900, 76.2400],
            'OutdoorScene': [74.4600, 78.9100, 79.6700, 81.4500, 82.9100, 83.0600, 83.1200, 83.7800, 83.3000, 85.3300],
            'ESP-Game': [83.6400, 86.8800, 87.2800, 87.5000, 88.0300, 88.7500, 88.5700, 89.0500, 89.8100, 90.1200],
            'IAPR-TC12': [62.0500, 64.8300, 64.0700, 64.8200, 67.4900, 70.1600, 72.2200, 73.0400, 73.2600, 74.3200],
        },
        'CurvFlow': {
            'ALOI': [67.8523, 72.6109, 71.7737, 72.1860, 71.7676, 75.7049, 74.8571, 75.7165, 76.8475, 75.5844],
            'Caltech7': [92.3517, 94.2986, 95.7063, 96.2850, 96.5973, 96.9797, 97.0177, 97.4661, 97.5339, 97.7446],
            'NUSWIDE20K': [65.4158, 69.3056, 71.1188, 72.1412, 72.9707, 73.4714, 73.7908, 74.0683, 74.2418, 74.5480],
            'OutdoorScene': [73.2576, 77.1062, 79.6937, 81.4140, 80.8333, 82.2742, 83.3677, 82.1823, 82.3259, 82.6190],
            'ESP-Game': [58.1220, 60.4975, 61.3522, 62.2252, 62.4976, 62.7868, 63.2441, 63.3666, 63.6657, 63.8049],
            'IAPR-TC12': [51.4672, 54.0076, 55.1738, 56.0089, 56.4602, 56.4575, 56.7293, 56.9707, 57.2553, 57.4745],
        },
        'GTMC-HOA': {
            'ALOI': [97.1296, 97.9630, 98.6111, 98.7963, 98.9815, 98.9815, 99.0741, 99.5370, 99.5370, 99.5370],
            'Caltech7': [90.3729, 91.4576, 93.6949, 94.6441, 95.1186, 95.7966, 96.3390, 97.1525, 96.8136, 96.9492],
            'NUSWIDE20K': [66.9200, 70.3650, 72.1550, 73.3550, 73.8250, 74.2800, 74.8500, 75.3100, 75.2500, 75.1900],
            'OutdoorScene': [77.2862, 80.2602, 79.4796, 83.8662, 85.2416, 84.8327, 86.2825, 87.0632, 86.8401, 87.7323],
            'ESP-Game': [84.9116, 87.4037, 88.3552, 88.8990, 89.2614, 89.7689, 90.0498, 90.2764, 90.2130, 90.3489],
            'IAPR-TC12': [65.4360, 68.8861, 70.2992, 71.4959, 72.6034, 73.5455, 73.2400, 74.0929, 74.2966, 74.4494],
        },
        'MDGG': {
            'ALOI': [78.3415, 83.6739, 82.4864, 84.8837, 84.4994, 84.3179, 83.8000, 83.2767, 82.9831, 82.6716],
            'Caltech7': [43.0450, 53.2579, 73.0726, 77.8287, 81.9910, 83.7209, 84.6138, 86.2443, 89.8397, 91.7391],
            'NUSWIDE20K': [42.6295, 51.0344, 56.5741, 58.7150, 61.1307, 62.8271, 64.3385, 65.0350, 65.5364, 66.0040],
            'OutdoorScene': [62.6312, 68.4746, 74.4595, 78.1116, 79.6528, 80.7439, 81.3975, 80.8803, 81.9621, 80.3571],
            'ESP-Game': [52.1023, 66.3612, 72.5477, 78.7242, 79.5116, 79.9145, 80.4826, 80.6981, 80.8868, 81.0954],
            'IAPR-TC12': [42.1790, 51.1869, 56.5638, 60.2355, 63.0657, 66.2605, 67.2307, 68.7715, 69.4722, 69.9236],
        },
        'MvDHP': {
            'ALOI': [98.0088, 98.5648, 98.5012, 98.8064, 98.7784, 99.6308, 99.5960, 99.7778, 99.7521, 99.8140],
            'Caltech7': [92.2586, 94.8136, 96.1991, 96.1240, 96.7258, 97.3815, 97.4631, 97.4457, 97.5602, 97.4237],
            'NUSWIDE20K': [68.8341, 72.5300, 74.1333, 74.9129, 75.9385, 76.4283, 76.6873, 76.5340, 76.9444, 76.9950],
            'OutdoorScene': [76.0122, 81.7751, 82.4876, 83.6769, 85.5003, 86.1757, 85.6757, 86.0818, 86.2758, 85.3346],
            'ESP-Game': [83.3444, 87.4213, 88.7417, 89.1467, 89.9177, 90.1269, 90.2456, 90.2665, 90.6970, 90.9746],
            'IAPR-TC12': [66.1246, 70.7590, 71.9701, 72.9260, 73.8046, 74.6045, 74.8589, 75.4899, 75.5373, 75.4757],
        },
    },
    'F1': {
        'DUANet': {
            'ALOI': [53.4252, 91.9723, 95.1951, 96.6817, 97.3968, 96.9800, 97.4039, 97.7877, 98.1837, 98.3812],
            'Caltech7': [47.8781, 55.6014, 62.7284, 65.1005, 64.8235, 66.0687, 65.4646, 66.2337, 68.0398, 68.6343],
            'NUSWIDE20K': [28.1047, 30.1206, 31.7572, 33.1236, 34.0833, 34.9622, 35.6069, 36.2259, 36.7034, 36.9985],
            'OutdoorScene': [54.9280, 59.0662, 62.1392, 63.9554, 64.8057, 66.0683, 66.5951, 66.3731, 67.8229, 69.0944],
            'ESP-Game': [43.8791, 47.4478, 48.6026, 48.7234, 49.8093, 50.1441, 50.7732, 51.1871, 51.3408, 52.0557],
            'IAPR-TC12': [35.3898, 37.3405, 39.6837, 41.4038, 42.1345, 41.6606, 42.8874, 43.8024, 43.9289, 44.0116],
        },
        'MMdynamics': {
            'ALOI': [39.6335, 45.6757, 48.3585, 27.1831, 55.8408, 56.0609, 46.1720, 56.2903, 49.1813, 56.5404],
            'Caltech7': [28.9370, 33.6517, 33.7732, 33.5167, 33.6577, 34.2473, 34.9910, 33.2848, 33.9353, 34.0603],
            'NUSWIDE20K': [59.6804, 61.6533, 63.0560, 63.6806, 64.0854, 64.5129, 64.8344, 64.8423, 65.1733, 65.3109],
            'OutdoorScene': [52.0272, 55.7188, 62.2111, 64.8236, 66.3548, 67.2476, 67.9945, 68.7472, 69.1324, 69.4712],
            'ESP-Game': [66.6812, 68.1087, 70.0507, 70.9535, 71.5800, 72.0157, 72.1695, 72.2563, 72.3876, 72.4438],
            'IAPR-TC12': [46.3954, 50.0335, 52.0378, 52.9969, 53.8711, 54.9180, 55.1791, 55.6385, 56.2542, 56.6254],
        },
        'LGCNFF': {
            'ALOI': [90.5640, 95.3900, 96.2600, 96.9240, 97.4900, 97.5040, 97.3220, 97.4240, 97.1660, 97.4840],
            'Caltech7': [58.0280, 62.6980, 71.0820, 72.4980, 71.7820, 70.0820, 73.8560, 73.1280, 70.2320, 72.9220],
            'NUSWIDE20K': [22.8400, 23.3040, 22.5400, 25.4680, 24.2580, 24.7920, 19.4720, 20.5960, 19.2640, 20.1280],
            'OutdoorScene': [72.0380, 76.5700, 77.7660, 78.8760, 79.1200, 79.6200, 79.7020, 79.9960, 80.0120, 80.0020],
            'ESP-Game': [59.7360, 68.7240, 65.5180, 65.4520, 67.6480, 68.3080, 69.7480, 70.2340, 64.6900, 68.7100],
            'IAPR-TC12': [51.9220, 57.4000, 59.3440, 60.8040, 61.6620, 61.9720, 62.9320, 63.2140, 63.3840, 63.8900],
        },
        'PDMF': {
            'ALOI': [13.4761, 12.3737, 17.6304, 14.8192, 21.1700, 32.3230, 18.0529, 30.8578, 23.8396, 28.8732],
            'Caltech7': [12.9059, 24.1491, 23.7280, 23.4024, 21.1657, 24.6343, 20.3248, 21.8089, 20.3986, 21.8893],
            'NUSWIDE20K': [28.3520, 53.9621, 54.1222, 56.6695, 64.8660, 73.1232, 75.4621, 73.5838, 75.8531, 76.0955],
            'OutdoorScene': [65.0553, 59.0352, 63.6367, 57.5693, 55.4585, 52.4427, 49.1147, 68.1071, 71.8372, 71.6518],
            'ESP-Game': [37.0142, 67.7440, 58.4106, 63.7741, 64.0095, 63.2533, 67.0687, 72.1092, 73.7066, 81.4888],
            'IAPR-TC12': [47.8718, 48.4396, 59.2195, 62.1597, 61.3317, 60.8183, 61.8253, 55.4319, 63.8598, 63.9818],
        },
        'RCML': {
            'ALOI': [44.9349, 82.8087, 82.4089, 94.9235, 94.2594, 94.3256, 93.8356, 97.7911, 97.7695, 97.5958],
            'Caltech7': [71.6879, 65.2415, 68.2662, 74.1404, 75.5889, 71.1772, 86.1736, 86.5222, 88.9113, 89.7045],
            'NUSWIDE20K': [66.2089, 70.8979, 72.1593, 72.2948, 73.0268, 73.5541, 73.8895, 74.2490, 74.1953, 74.3495],
            'OutdoorScene': [70.3679, 74.1471, 76.3716, 78.7503, 79.3580, 78.5800, 80.1301, 80.1318, 80.2418, 82.0764],
            'ESP-Game': [80.7213, 82.8804, 84.6600, 84.0810, 86.3617, 86.8191, 87.7538, 87.8385, 88.0060, 88.7945],
            'IAPR-TC12': [57.5056, 62.7741, 66.7769, 68.3201, 69.2204, 70.2431, 70.9924, 70.8909, 70.6013, 71.8602],
        },
        'KAMSSM': {
            'ALOI': [87.5100, 94.7500, 95.0100, 95.0500, 93.0400, 95.8400, 83.5600, 96.8800, 96.2900, 97.4600],
            'Caltech7': [64.5900, 75.9400, 76.7100, 80.3800, 83.4100, 87.0200, 88.0300, 88.6000, 88.2200, 88.6900],
            'NUSWIDE20K': [65.1100, 65.4600, 69.3800, 72.3600, 74.1700, 75.1100, 75.7000, 76.1100, 76.1900, 76.1400],
            'OutdoorScene': [74.5400, 78.9800, 79.7900, 81.5900, 82.9900, 83.1500, 83.2100, 83.8900, 83.5100, 85.4200],
            'ESP-Game': [83.4600, 86.5200, 86.9200, 87.1400, 87.6800, 88.3800, 88.2400, 88.7000, 89.4700, 89.7900],
            'IAPR-TC12': [63.1900, 65.7900, 65.0500, 65.6800, 68.2700, 71.0200, 72.9700, 73.9300, 74.1200, 75.1900],
        },
        'CurvFlow': {
            'ALOI': [63.1982, 69.1102, 68.6840, 69.2291, 68.9584, 73.5337, 72.6174, 73.4513, 74.4138, 73.0221],
            'Caltech7': [68.3177, 76.6987, 82.7621, 83.7980, 84.8413, 86.9293, 86.0943, 88.4035, 89.1472, 89.7311],
            'NUSWIDE20K': [65.2616, 69.1635, 71.0142, 72.0326, 72.8631, 73.3573, 73.6687, 73.9559, 74.1287, 74.4378],
            'OutdoorScene': [73.3592, 77.2176, 80.1339, 81.8400, 81.2901, 82.6658, 83.6953, 82.5771, 82.6996, 82.9676],
            'ESP-Game': [57.3185, 59.7868, 60.6402, 61.5417, 61.9027, 62.1953, 62.6638, 62.8047, 63.0869, 63.2406],
            'IAPR-TC12': [51.2455, 53.8336, 55.1247, 56.0424, 56.5174, 56.5548, 56.8324, 57.1154, 57.3704, 57.5891],
        },
        'GTMC-HOA': {
            'ALOI': [97.1566, 97.9853, 98.6371, 98.8025, 98.9963, 98.9813, 99.0742, 99.5662, 99.5368, 99.5368],
            'Caltech7': [60.8200, 67.2266, 76.7819, 79.4879, 81.7780, 84.6136, 86.1291, 88.4216, 87.8485, 88.4938],
            'NUSWIDE20K': [66.4809, 70.0217, 71.9869, 73.0899, 73.6618, 74.1400, 74.7019, 75.2024, 75.1470, 75.1752],
            'OutdoorScene': [77.9350, 80.6755, 79.6662, 84.0902, 85.4260, 85.0324, 86.4110, 87.2810, 87.0089, 87.9193],
            'ESP-Game': [84.5426, 87.1334, 88.0467, 88.5964, 88.9597, 89.4241, 89.7306, 90.0290, 89.9117, 90.0933],
            'IAPR-TC12': [66.5034, 70.0402, 71.3556, 72.5486, 73.7759, 74.5425, 74.2691, 75.0999, 75.2830, 75.4845],
        },
        'MDGG': {
            'ALOI': [77.4860, 83.8688, 81.9167, 85.4879, 84.8881, 84.6029, 84.0773, 83.2273, 82.7038, 82.6020],
            'Caltech7': [38.1969, 48.0068, 60.6065, 65.4833, 69.7180, 70.1821, 71.4683, 73.2187, 76.7350, 78.7497],
            'NUSWIDE20K': [42.7621, 51.1766, 56.5015, 58.4638, 61.0079, 62.5886, 64.0831, 64.7104, 65.2189, 65.7189],
            'OutdoorScene': [62.7711, 68.2162, 74.3469, 78.4359, 80.0880, 80.9893, 81.5912, 81.0939, 82.1896, 80.4093],
            'ESP-Game': [52.0318, 66.6310, 72.4855, 78.5637, 79.3760, 79.7590, 80.3232, 80.5649, 80.7315, 80.9371],
            'IAPR-TC12': [42.7059, 51.8655, 57.3914, 61.0493, 63.9394, 67.1145, 67.9808, 69.5506, 70.1167, 70.3590],
        },
        'MvDHP': {
            'ALOI': [98.0105, 98.5663, 98.5018, 98.8058, 98.7783, 99.6317, 99.5963, 99.7778, 99.7526, 99.8139],
            'Caltech7': [69.8087, 79.4610, 83.0518, 83.1477, 85.7964, 87.9049, 87.7292, 88.6666, 88.9586, 86.8727],
            'NUSWIDE20K': [68.5534, 72.2869, 74.0094, 74.7487, 75.8247, 76.2889, 76.4555, 76.3693, 76.7496, 76.8334],
            'OutdoorScene': [76.1748, 82.0772, 82.7130, 83.9494, 85.7459, 86.3782, 85.9275, 86.2482, 86.3980, 85.5089],
            'ESP-Game': [83.1173, 87.1618, 88.4684, 88.8518, 89.6550, 89.8493, 89.9752, 89.9853, 90.4200, 90.6795],
            'IAPR-TC12': [67.3129, 71.8043, 72.9180, 74.0126, 74.7251, 75.4867, 75.7919, 76.4346, 76.4631, 76.3661],
        },
    },
}

markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "d", "P", "8"]

colors = [
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]
OUR_METHOD = "MvDHP"
OUR_COLOR = "#D55E00"


def _validate_acc_plot_data():
    """Fail early if ACC plotting order or curve data no longer match labels."""
    acc_method_order = list(data["ACC"])
    if method_order != acc_method_order:
        raise ValueError(
            "method_order must exactly match the ACC data order. "
            f"Expected {acc_method_order}, got {method_order}."
        )

    for method in method_order:
        for dataset_name in datasets:
            values = data["ACC"][method].get(dataset_name)
            if values is None or len(values) != len(ratio_labels):
                raise ValueError(
                    f"Invalid ACC curve for {method} on {dataset_name}: "
                    f"expected {len(ratio_labels)} supervision-ratio values."
                )


def _normalize_metric(metric):
    metric = metric.upper()
    if metric not in data:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {', '.join(data)}")
    return metric


def _safe_filename(text):
    return text.replace("/", "_").replace(" ", "_")


def _save_figure(fig, save_path, *, bbox_inches="tight", pad_inches=0.01):
    """Export each figure as editable PDF and SVG files."""
    file_stem, _ = os.path.splitext(save_path)
    for extension in ("pdf", "svg"):
        output_path = f"{file_stem}.{extension}"
        fig.savefig(
            output_path,
            format=extension,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        print(f"Figure saved as {output_path}")


def _row_major_legend_entries(handles, labels, ncol):
    """Arrange legend entries left-to-right by row for a multi-column legend."""
    nrows = (len(handles) + ncol - 1) // ncol
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(handles)
    ]
    return [handles[i] for i in order], [labels[i] for i in order]


def plot_single_dataset(dataset_name, save_path=None, metric="ACC", with_legend=False):
    metric = _normalize_metric(metric)
    metric_data = data[metric]
    metric_label = metric_labels[metric]

    plt.figure(figsize=(5.2, 3.6))

    for idx, method in enumerate(method_order):
        y = metric_data[method][dataset_name]
        x = ratio_labels[:len(y)]

        plt.plot(
            x,
            y,
            marker=markers[idx % len(markers)],
            color=OUR_COLOR if method == OUR_METHOD else colors[idx % len(colors)],
            label=method,
            linewidth=2.2 if method == OUR_METHOD else 1.6,
            markersize=5.5 if method == OUR_METHOD else 4.5,
            zorder=4 if method == OUR_METHOD else 3,
        )

    plt.xlabel("Supervision Ratio (%)", fontsize=20)
    plt.ylabel(f"{metric_label} (%)", fontsize=20)
    plt.xticks(ratio_labels, fontsize=9)
    plt.yticks(fontsize=9)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    if with_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles, labels = _row_major_legend_entries(handles, labels, ncol=6)
        plt.legend(
            handles,
            labels,
            fontsize=8,
            ncol=6,
            loc="best",
            frameon=False,
        )

    plt.tight_layout(pad=0.25)

    if save_path:
        _save_figure(plt.gcf(), save_path)

    plt.close()


def plot_legend_only(save_path="legend.pdf"):
    fig, ax = plt.subplots(figsize=(13.5, 1.5))

    for idx, method in enumerate(method_order):
        ax.plot(
            [],
            [],
            marker=markers[idx % len(markers)],
            color=OUR_COLOR if method == OUR_METHOD else colors[idx % len(colors)],
            label=method,
            linewidth=2.2 if method == OUR_METHOD else 1.6,
            markersize=5.5 if method == OUR_METHOD else 4.5,
        )

    ax.axis("off")

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _row_major_legend_entries(handles, labels, ncol=6)

    legend = ax.legend(
        handles,
        labels,
        loc="center",
        ncol=6,
        fontsize=8.5,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.75,
        handletextpad=0.3,
        borderaxespad=0.0,
        borderpad=0.0,
        labelspacing=0.2,
    )

    fig.canvas.draw()

    bbox = legend.get_window_extent(
        fig.canvas.get_renderer()
    ).transformed(fig.dpi_scale_trans.inverted())

    _save_figure(fig, save_path, bbox_inches=bbox, pad_inches=0.003)
    plt.close(fig)


def plot_all_subplots(selected_datasets=None, save_path=None, metric="ACC"):
    metric = _normalize_metric(metric)
    metric_data = data[metric]

    if selected_datasets is None:
        selected_datasets = datasets

    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = list(axes.flat)

    for i, ds_name in enumerate(selected_datasets):
        ax = axes[i]

        for idx, method in enumerate(method_order):
            y = metric_data[method][ds_name]
            x = ratio_labels[:len(y)]

            ax.plot(
                x,
                y,
                marker=markers[idx % len(markers)],
                color=OUR_COLOR if method == OUR_METHOD else colors[idx % len(colors)],
                label=method,
                linewidth=2.0 if method == OUR_METHOD else 1.5,
                markersize=5 if method == OUR_METHOD else 4,
                zorder=4 if method == OUR_METHOD else 3,
            )

        ax.set_xlabel("Supervision Ratio (%)", fontsize=11)
        ax.set_ylabel(f"{metric_labels[metric]} (%)", fontsize=11)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_title(f"({chr(97 + i)}) {ds_name}", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xticks(ratio_labels)
        ax.tick_params(axis="both", labelsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _row_major_legend_entries(handles, labels, ncol=6)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        handlelength=1.4,
        columnspacing=0.75,
        handletextpad=0.3,
        borderaxespad=0.0,
        borderpad=0.0,
        labelspacing=0.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        _save_figure(fig, save_path)

    plt.close(fig)


if __name__ == "__main__":
    _validate_acc_plot_data()

    output_root = "label_ratio_pdf"
    os.makedirs(output_root, exist_ok=True)

    for metric in ("ACC",):
        metric_dir = os.path.join(output_root, metric.lower())
        os.makedirs(metric_dir, exist_ok=True)

        for ds in datasets:
            file_name = f"{metric.lower()}_{_safe_filename(ds)}_results.pdf"
            save_path = os.path.join(metric_dir, file_name)

            plot_single_dataset(
                dataset_name=ds,
                save_path=save_path,
                metric=metric,
                with_legend=False,
            )

        plot_all_subplots(
            save_path=os.path.join(metric_dir, f"{metric.lower()}_all_datasets_subplots.pdf"),
            metric=metric,
        )

    plot_legend_only(
        save_path=os.path.join(output_root, "legend.pdf")
    )
