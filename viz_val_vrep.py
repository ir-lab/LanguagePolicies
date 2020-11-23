# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
from utils.voice import Voice

# Where to find the results file?
FILE_PATH = "ours_full_cl.json"
# Where to find the normalization
NORM_PATH = "../GDrive/normalization_v2.pkl"

normalization = pickle.load(open(NORM_PATH, mode="rb"), encoding="latin1")
norm          = np.take(normalization["values"], indices=[0,1,2,3,4,5,30], axis=1)
voice_class   = Voice(load=False)

def normalize(value, v_min, v_max):
    if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
        len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
        raise ArrayDimensionMismatch()
    value = np.copy(value)
    v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
    v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
    value = (value - v_min) / (v_max - v_min)
    return value

def rotateCoordinates(px, py, angle=-45):
    r_px  = px * np.cos(np.deg2rad(angle)) + py * np.sin(np.deg2rad(angle))
    r_py  = py * np.cos(np.deg2rad(angle)) - px * np.sin(np.deg2rad(angle))
    return (r_px, r_py)

def transformCoordinate(x, y, z=-1.6260e-02):
    # top_left:     [+1.0097e+00, -6.2800e-01, +5.0042e-03] [36,  3  ] [-5.3933e-01, +8.8861e-01, -1.6260e-02]
    # bottom_left:  [+1.0097e+00, +6.2850e-01, +5.0042e-03] [36,  317] [-1.4278e+00, +1.2640e-04, -1.6189e-02]
    # bottom_right: [-1.0100e+00, +6.2850e-01, +5.0042e-03] [539, 317] [+3.3816e-04, -1.4280e+00, -1.6189e-02]
    # top_right:    [-1.0100e+00, -6.2800e-01, +5.0042e-03] [539, 3  ] [+8.8882e-01, -5.3954e-01, -1.6260e-02]
    
    # source = [[-5.3933e-01, +8.8861e-01, -1.6260e-02], [-1.4278e+00, +1.2640e-04, -1.6189e-02], [+3.3816e-04, -1.4280e+00, -1.6189e-02], [+8.8882e-01, -5.3954e-01, -1.6260e-02]]
    # destination = [[36,3], [36,317], [539,317], [539,3]]
    # np.linalg.solve(source[:3], destination[:3])
    tns = [[   175.39848388,   -176.86537052], [  -176.80979217,   -176.86682693], [-17694.50268915,  -3983.8149321 ]]
    return np.dot([x,y,z], tns)

def plotPhaseArrows(data, key):
    data     = data[key]
    ex_names = data.keys()

    fig = plt.figure(figsize=(5.69, 3.2))
    plt.imshow(plt.imread("vrep_empty.png"))
    for exp in ex_names:
        if key == "phase_1":
            color = "g" if data[exp]["success"] else "r"
        else:
            color = "g" if data[exp]["success"] == 2 else ("y" if data[exp]["success"] == 1 else "r")
        x,y   = transformCoordinate(*data[exp]["locations"]["current"])
        x2,y2 = transformCoordinate(*data[exp]["locations"]["target"])       

        plt.arrow(x, y, x2-x, y2-y, length_includes_head=True, width=0.01, head_width=2, color=color)
   
    plt.title("Success Rate: " + key)

def averageBallSuccessPerTask(data):
    ball_arrays = [data["phase_2"][exp]["ball_array"] for exp in data["phase_2"].keys()]
    success_rate = []
    for array in ball_arrays:
        if len(array) == 0:
            success_rate.append(0)
        else:
            success_rate.append(float(np.sum(array))/float(len(array)))
    return np.mean(success_rate)

def interpolateTrajectory(trj, target):
    trj            = np.asarray(trj)
    current_length = trj.shape[0]
    dimensions     = trj.shape[1]
    result         = np.zeros((target, trj.shape[1]), dtype=np.float32)
            
    for i in range(dimensions):
        result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
    
    return result

def calcMAE(trj, states):
    trj = np.asarray(trj)
    stt = np.asarray(states)
    stt = interpolateTrajectory(stt, trj.shape[0])
    return mean_absolute_error(trj[:,:6], stt[:,:6]) 

def getUsedFeatures(voice):
    features    = [0,0,0] # color, size, shape
    color_words = voice_class.synonyms["red"] + voice_class.synonyms["green"] + voice_class.synonyms["blue"] + voice_class.synonyms["yellow"] + voice_class.synonyms["pink"] + [voice_class.test_words["red"], voice_class.test_words["green"], voice_class.test_words["blue"], voice_class.test_words["yellow"], voice_class.test_words["pink"]]
    size_words  = voice_class.synonyms["small"] + voice_class.synonyms["large"] + [voice_class.test_words["small"], voice_class.test_words["large"]]
    shape_words = voice_class.synonyms["round"] + voice_class.synonyms["square"] + [voice_class.test_words["round"], voice_class.test_words["square"]]

    for word in color_words:
        if word in voice:
            features[0] = 1
    for word in size_words:
        if word in voice:
            features[1] = 2
    for word in shape_words:
        if word in voice:
            features[2] = 4
    return features

def debugColors(voice):
    features    = [0,0,0,0,0] # color, size, shape
    red    = voice_class.synonyms["red"] + [voice_class.test_words["red"]]
    green  = voice_class.synonyms["green"] + [voice_class.test_words["green"]]
    blue   = voice_class.synonyms["blue"] + [voice_class.test_words["blue"]]
    yellow = voice_class.synonyms["yellow"] + [voice_class.test_words["yellow"]]
    pink   = voice_class.synonyms["pink"] + [voice_class.test_words["pink"]]

    for word in red:
        if word in voice:
            features[0] = 1
    for word in green:
        if word in voice:
            features[1] = 2
    for word in blue:
        if word in voice:
            features[2] = 4
    for word in yellow:
        if word in voice:
            features[3] = 8
    for word in pink:
        if word in voice:
            features[4] = 16
    return features

def overallStatistics(data):
    data = cleanJson(data)
    num_tasks = len(data["phase_1"].keys())

    p1_correct_objects = [name for name in data["phase_1"].keys() if data["phase_1"][name]["locations"]["tid"] == data["phase_1"][name]["locations"]["tid/actual"][0][0]]
    p2_correct_objects = [name for name in data["phase_2"].keys() if data["phase_2"][name]["locations"]["tid"] == data["phase_2"][name]["locations"]["tid/actual"][0][1]]

    p1_names   = [name for name in data["phase_1"].keys() if data["phase_1"][name]["success"] and name in p1_correct_objects]
    p1_success = len(p1_names)
    p2_names   = [name for name in data["phase_2"].keys() if (lambda array: False if len(array) == 0 else float(np.sum(array))/float(len(array)) > 0.5)(data["phase_2"][name]["ball_array"]) and name in p2_correct_objects]
    p2_success = len(p2_names)
    oa_success = len([name for name in p1_names if name in p2_names])

    print("Failed picking sentences:")
    for stn in [(data["phase_1"][name]["language"]["original"], name) for name in data["phase_1"].keys() if name not in p1_names]:
        print("  -> {}\t{}".format(stn[1], stn[0]))
    print("Failed pouring sentences:")
    for stn in [(data["phase_2"][name]["language"]["original"], name) for name in data["phase_2"].keys() if name not in p2_names]:
        print("  -> {}\t{}".format(stn[1], stn[0]))

    odist_success = np.mean(
        [data["phase_1"][name]["locations"]["distance"] for name in data["phase_1"].keys() if name in p1_names] +
        [data["phase_2"][name]["locations"]["distance"] for name in data["phase_2"].keys() if name in p2_names]
    )
    odist_failure = np.mean(
        [data["phase_1"][name]["locations"]["distance"] for name in data["phase_1"].keys() if name not in p1_names] +
        [data["phase_2"][name]["locations"]["distance"] for name in data["phase_2"].keys() if name not in p2_names]
    )

    if p2_success + p1_success == num_tasks * 2:
        odist_failure = 0

    cdetect = 0
    cdetect += len([name for name in data["phase_1"].keys() if data["phase_1"][name]["locations"]["tid"] == data["phase_1"][name]["locations"]["tid/actual"][0][0]])
    cdetect += len(p2_correct_objects)
    cdetect = float(cdetect)/float(2*num_tasks) * 100.0

    ball_arrays = [[len(data["phase_2"][name]["ball_array"]), np.sum(data["phase_2"][name]["ball_array"]), data["phase_2"][name]["language"]["quantity"], name] for name in data["phase_2"].keys()]
    max_balls   = max(ball_arrays, key=lambda x: x[0])[0]
    avg_small   = [float(v[1])/float(max_balls) for v in ball_arrays if v[2] == 1]
    avg_large   = [float(v[1])/float(max_balls) for v in ball_arrays if v[2] == 2]

    avg_small_success = np.sum([1 for v in ball_arrays if v[2] == 1 and abs(float(v[1])/float(max_balls) - np.mean(avg_small)) < abs(float(v[1])/float(max_balls) - np.mean(avg_large)) and v[3] in p1_names]) / float(len(avg_small))
    avg_large_success = np.sum([1 for v in ball_arrays if v[2] == 2 and abs(float(v[1])/float(max_balls) - np.mean(avg_small)) > abs(float(v[1])/float(max_balls) - np.mean(avg_large)) and v[3] in p2_names]) / float(len(avg_large))
    
    s_d0 = [name for name in data["phase_2"].keys() if data["phase_2"][name]["language"]["features"] == 0]
    s_d0 = np.nan if len(s_d0) == 0 else len([name for name in s_d0 if name in p2_names]) / float(len(s_d0))

    s_d1 = [name for name in data["phase_2"].keys() if data["phase_2"][name]["language"]["features"] == 1]
    s_d1 = np.nan if len(s_d1) == 0 else len([name for name in s_d1 if name in p2_names]) / float(len(s_d1))

    s_d2 = [name for name in data["phase_2"].keys() if data["phase_2"][name]["language"]["features"] == 2]
    s_d2 = np.nan if len(s_d2) == 0 else len([name for name in s_d2 if name in p2_names]) / float(len(s_d2))

    s_d3 = [name for name in data["phase_2"].keys() if data["phase_2"][name]["language"]["features"] == 3]
    s_d3 = np.nan if len(s_d3) == 0 else len([name for name in s_d3 if name in p2_names]) / float(len(s_d3))   

    a_d0 = [name for name in data["phase_1"].keys() if data["phase_1"][name]["language"]["features"] == 0]
    a_d0 = np.nan if len(a_d0) == 0 else len([name for name in a_d0 if name in p1_names]) / float(len(a_d0))

    a_d1 = [name for name in data["phase_1"].keys() if data["phase_1"][name]["language"]["features"] == 1]
    a_d1 = np.nan if len(a_d1) == 0 else len([name for name in a_d1 if name in p1_names]) / float(len(a_d1)) 
    
    if "trajectory" in data["phase_1"][list(data["phase_1"].keys())[0]]:
        mae_p1 = np.mean([calcMAE(data["phase_1"][name]["trajectory"]["gt"], data["phase_1"][name]["trajectory"]["state"]) for name in data["phase_1"].keys()])
        mae_p2 = np.mean([calcMAE(data["phase_2"][name]["trajectory"]["gt"], data["phase_2"][name]["trajectory"]["state"]) for name in data["phase_2"].keys()])
    else:
        mae_p1 = np.nan
        mae_p2 = np.nan

    vfeatures_total = np.sum([getUsedFeatures(data["phase_2"][name]["language"]["original"]) for name in data["phase_2"].keys()], axis=1)
    if len(p2_names) == 0:
        vfeatures_pass = [[]]
    else:
        vfeatures_pass  = np.sum([getUsedFeatures(data["phase_2"][name]["language"]["original"]) for name in data["phase_2"].keys() if name in p2_names], axis=1)
    unique, counts  = np.unique(vfeatures_total, return_counts=True)
    vfeatures_total = dict(zip(unique, counts))
    unique, counts  = np.unique(vfeatures_pass, return_counts=True)
    vfeatures_pass  = dict(zip(unique, counts))

    vfeature_names  = {0: "none", 1: "color", 2: "size", 3: "color + size", 4:"shape", 5:"color + shape", 6:"size + shape", 7:"color + size + shape"}
    vfeature_success = {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4:np.nan, 5:np.nan, 6:np.nan, 7:np.nan}
    for i in range(8):
        if i in vfeatures_total.keys():
            if i in vfeatures_pass.keys():
                vfeature_success[i] = float(vfeatures_pass[i])/float(vfeatures_total[i]) 
            else:
                vfeature_success[i] = 0.0    

    print("Overall results:")
    print("  Pickup:      {}/{} ({:.1f}%) successful".format(p1_success, num_tasks, float(p1_success)/num_tasks * 100.0))
    print("  Pouring:     {}/{} ({:.1f}%) successful".format(p2_success, num_tasks, float(p2_success)/num_tasks * 100.0))
    print("  Consecutive: {}/{} ({:.1f}%) consecutive tasks successful".format(oa_success, num_tasks, float(oa_success)/num_tasks * 100.0))
    print("  Objects:     {:.1f}%".format(cdetect))
    print("  Avg inside:  {:.1f}% of dropped balls".format(averageBallSuccessPerTask(data) * 100.0))
    print("  'few'%:      {:.1f}%".format(100*np.mean(avg_small)))
    print("  'many'%:     {:.1f}%".format(100*np.mean(avg_large)))
    print("  D0 success:  {:.1f}%".format(s_d0*100))
    print("  D1 success:  {:.1f}%".format(s_d1*100))
    print("  D2 success:  {:.1f}%".format(100*s_d2))
    print("  D3 success:  {:.1f}%".format(100*s_d3))
    print("  Distance:    {:.1f}cm".format((odist_failure - odist_success) * 100))
    print("  Detailed Features:")
    for i in range(8):
        print("    {}: {:.1f}% out of {}".format(vfeature_names[i], vfeature_success[i] * 100, 0 if i not in vfeatures_total.keys() else vfeatures_total[i]))

    print("Latex:")
    print("{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} ".format(
        float(p1_success)/num_tasks,
        float(p2_success)/num_tasks,
        float(oa_success)/num_tasks,
        cdetect / 100.0,
        averageBallSuccessPerTask(data),
        (avg_small_success + avg_large_success) / 2.0,
        (mae_p1 + mae_p2)/2.0,
        ((odist_failure + odist_success) / 2.0 ) * 100,
        vfeature_success[0],
        vfeature_success[1],
        vfeature_success[2],
        vfeature_success[4],
        vfeature_success[3],
        vfeature_success[5],
        vfeature_success[6],
        vfeature_success[7]
        ))

def cleanJson(data):
    def removeEmpty(phs):
        delete = []
        for key in data[phs].keys():
            if "locations" not in data[phs][key].keys():
                delete.append(key)
        for key in delete:
            data[phs].pop(key, None)
    removeEmpty("phase_1")
    removeEmpty("phase_2")
    return data

if __name__ == "__main__":
    path = FILE_PATH

    with open(path, "r") as fh:
        data = json.load(fh)

    print(data.keys())
    overallStatistics(data)

