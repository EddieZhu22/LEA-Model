# -*- coding:utf-8 -*-
##############################################################
# Created Date: Thursday, August 31st 2023
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def lst_equal_split(lst: np.array, n: int) -> list:
    """equally split a list into n size chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_data(dat_lst: list, title: str = "TEST") -> plt:
    fig, ax = plt.subplots()
    plt.plot(dat_lst, label='mobility change', color='b')
    plt.xticks(rotation=90)
    # ax.set_xticks(df_grocery_and_pharmacy["date"].map(lambda x: str(x)))
    plt.title(title)
    plt.show()


def find_first_wave_t0_t2_t4(state_values_tuple: tuple[list], time_window: int = 7) -> dict:
    # the input lst_state provide valid data, e.g. for 140 days, but not all values

    lst_state = state_values_tuple[1]
    lst_state_dropna = pd.DataFrame(lst_state).dropna()[0].values.tolist()
    state_name = state_values_tuple[0]

    # split the list into equal 7 days, tw: time window
    lst_tw = list(lst_equal_split(np.array(lst_state_dropna), time_window))

    # calculate the mean of each 7 days
    lst_tw_mean_first_wave = [np.mean(i) for i in lst_tw]

    # show the plot of 7 days mean
    plot_data(lst_tw_mean_first_wave, title=state_name)

    # decide the curve is convex or concave: y = ax^2 + bx + c
    a = np.polyfit(list(range(len(lst_tw_mean_first_wave))),
                   lst_tw_mean_first_wave, 2)[0]

    print(f"{state_name}, a: ", a)

    if a > 0:
        # concave curve

        # find the index of minimum value of first wave: t2
        lst_tw_mean_first_wave_t2_index = lst_tw_mean_first_wave.index(
            min(lst_tw_mean_first_wave[1:-1]))

        # find the index of maximum value of first wave : t0
        lst_tw_mean_first_wave_disruption = lst_tw_mean_first_wave[:lst_tw_mean_first_wave_t2_index]

        lst_tw_mean_first_wave_t0_index = lst_tw_mean_first_wave_disruption.index(
            max(lst_tw_mean_first_wave_disruption))

        # Find t4 after the minimum value of first wave: t4
        # the criteria is that the value less than the previous value
        lst_tw_mean_first_wave_recover = lst_tw_mean_first_wave[lst_tw_mean_first_wave_t2_index:]

        # t4 equal to the last value if no value less than the previous value
        lst_tw_mean_first_wave_t4_index = len(lst_tw_mean_first_wave_recover) - 1

        for i in range(len(lst_tw_mean_first_wave_recover) - 1):
            if lst_tw_mean_first_wave_recover[i + 1] < lst_tw_mean_first_wave_recover[i]:
                lst_tw_mean_first_wave_t4_index = i
                break
        lst_tw_mean_first_wave_t4_index += lst_tw_mean_first_wave_t2_index

        # We have find the index for t0, t2, t4 from 7 days mean
        # find the read index for original data
        first_wave_t0_max_value = lst_tw[lst_tw_mean_first_wave_t0_index].max()
        first_wave_t2_min_value = lst_tw[lst_tw_mean_first_wave_t2_index].min()
        first_wave_t4_max_value = lst_tw[lst_tw_mean_first_wave_t4_index].max()

    else:
        # convex curve

        # find the index of max value of first wave: t2
        lst_tw_mean_first_wave_t2_index = lst_tw_mean_first_wave.index(
            max(lst_tw_mean_first_wave[1:-1]))

        # find the index of maximum value of first wave : t0
        lst_tw_mean_first_wave_disruption = lst_tw_mean_first_wave[:lst_tw_mean_first_wave_t2_index]

        lst_tw_mean_first_wave_t0_index = lst_tw_mean_first_wave_disruption.index(
            min(lst_tw_mean_first_wave_disruption))

        # Find t4 after the minimum value of first wave: t4
        # the criteria is that the value less than the previous value
        lst_tw_mean_first_wave_recover = lst_tw_mean_first_wave[lst_tw_mean_first_wave_t2_index:]

        # t4 equal to the last value if no value less than the previous value
        lst_tw_mean_first_wave_t4_index = len(lst_tw_mean_first_wave_recover) - 1

        for i in range(len(lst_tw_mean_first_wave_recover) - 1):
            if lst_tw_mean_first_wave_recover[i + 1] > lst_tw_mean_first_wave_recover[i]:
                lst_tw_mean_first_wave_t4_index = i
                break
        lst_tw_mean_first_wave_t4_index += lst_tw_mean_first_wave_t2_index

        # We have find the index for t0, t2, t4 from 7 days mean
        # find the read index for original data
        first_wave_t0_max_value = lst_tw[lst_tw_mean_first_wave_t0_index].min()
        first_wave_t2_min_value = lst_tw[lst_tw_mean_first_wave_t2_index].max()
        first_wave_t4_max_value = lst_tw[lst_tw_mean_first_wave_t4_index].min()

    t0_index = lst_state.index(first_wave_t0_max_value)
    t2_index = lst_state.index(first_wave_t2_min_value)
    t4_index = lst_state.index(first_wave_t4_max_value, t2_index)

    if t0_index > t2_index or t2_index > t4_index:
        print()
        print(f"Error: for {state_name}, t0_index > t2_index or t2_index > t4_index")
        print()

    t0_2_4 = {"t0_index": t0_index, "t2_index": t2_index, "t4_index": t4_index,
              "t0_val": first_wave_t0_max_value,
              "t2_val": first_wave_t2_min_value,
              "t4_val": first_wave_t4_max_value}

    print(f"t0_2_4 index: {t0_index, t2_index, t4_index}")

    return t0_2_4


def find_first_wave_t1_t3(lst_state: list, t0_2_4: tuple) -> dict:

    # find t1 and t3
    lst_t1 = lst_state[t0_2_4[0]:t0_2_4[1]]
    lst_t3 = lst_state[t0_2_4[1]:t0_2_4[2]]

    # drop the nan value
    lst_t1_dropna = pd.DataFrame(lst_t1).dropna()[0].tolist()
    lst_t3_dropna = pd.DataFrame(lst_t3).dropna()[0].tolist()

    gap = 0
    t1_val = lst_t1_dropna[0]
    for i in range(len(lst_t1_dropna) - 1):
        if abs(lst_t1_dropna[i + 1] - lst_t1_dropna[i]) > gap:
            gap = abs(lst_t1_dropna[i + 1] - lst_t1_dropna[i])
            t1_val = lst_t1_dropna[i]
    t1_index = lst_state.index(t1_val)

    gap = 0
    t3_val = lst_t3_dropna[0]
    for i in range(len(lst_t3_dropna) - 1):
        if abs(lst_t3_dropna[i + 1] - lst_t3_dropna[i]) > gap:
            gap = abs(lst_t3_dropna[i + 1] - lst_t3_dropna[i])
            t3_val = lst_t3_dropna[i]

    t3_index = lst_state.index(t3_val)

    return {"t1_index": t1_index, "t3_index": t3_index,
            "t1_val": t1_val, "t3_val": t3_val}


def find_ts(state_values_tuple: tuple[list], time_window: int = 7) -> dict:
    # The state_values_tuple: (state_name, state_values in list, time in list)

    lst_state = state_values_tuple[1]
    lst_time = state_values_tuple[2]  # the time list from table

    t0_2_4 = find_first_wave_t0_t2_t4(state_values_tuple, time_window)

    t0_index = t0_2_4["t0_index"]
    t2_index = t0_2_4["t2_index"]
    t4_index = t0_2_4["t4_index"]

    t1_3 = find_first_wave_t1_t3(lst_state, (t0_index, t2_index, t4_index))

    first_wave = {**t0_2_4, **t1_3, **{"t0_time": lst_time[t0_index],
                                       "t1_time": lst_time[t1_3["t1_index"]],
                                       "t2_time": lst_time[t2_index],
                                       "t3_time": lst_time[t1_3["t3_index"]],
                                       "t4_time": lst_time[t4_index]}}
    return first_wave


if __name__ == '__main__':

    # park, residential
    # grocery, retail, transit, workplace

    # define mobility categories
    mobility_categories = ["retail_and_recreation_percent_change_from_baseline",
                           "grocery_and_pharmacy_percent_change_from_baseline",
                           "parks_percent_change_from_baseline",
                           "transit_stations_percent_change_from_baseline",
                           "workplaces_percent_change_from_baseline",
                           "residential_percent_change_from_baseline"]

    # Calculate T for each category each state
    path_each_category = [
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_grocery_and_pharmacy_percent_change_from_baseline.csv",
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_parks_percent_change_from_baseline.csv",
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_residential_percent_change_from_baseline.csv",
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_retail_and_recreation_percent_change_from_baseline.csv",
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_transit_stations_percent_change_from_baseline.csv",
        r"./data/Region_Mobility_report_CSVs/us_mobility_change_by_date_2020_workplaces_percent_change_from_baseline.csv"
    ]

    df_grocery_and_pharmacy = pd.read_csv(path_each_category[0])

    state_name_lst = df_grocery_and_pharmacy.columns[1:-3].to_list()
    time_lst = df_grocery_and_pharmacy["month_day"].to_list()

    res = []
    for state_name in state_name_lst:
        df_grocery_and_pharmacy_state = df_grocery_and_pharmacy[state_name].to_list(
        )
        first_wave = find_ts(
            (state_name, df_grocery_and_pharmacy_state[:141], time_lst[:141]), 7)
        print(state_name, first_wave)
        res.append(first_wave)

    df_mobility_first_wave = pd.DataFrame(res)
    df_mobility_first_wave["state"] = state_name_lst
    df_mobility_first_wave = df_mobility_first_wave[["state", "t0_time", "t0_index", "t0_val",
                                                     "t1_time", "t1_index", "t1_val",
                                                     "t2_time", "t2_index", "t2_val",
                                                     "t3_time", "t3_index", "t3_val",
                                                     "t4_time", "t4_index", "t4_val"]]

    for mobility_type in path_each_category[:1]:
        df_mobility = pd.read_csv(mobility_type)
        state_name_lst = df_mobility.columns[1:-3].to_list()
        time_lst = df_mobility["month_day"].to_list()

        res = []
        for state_name in state_name_lst:
            df_mobility_state = df_mobility[state_name].to_list()

            # for parks use 260 days, which mean that parks have no double queue

            if "parks" in mobility_type:
                first_wave = find_ts(
                    (state_name, df_mobility_state[:141], time_lst[:141]), 7)
            else:
                first_wave = find_ts(
                    (state_name, df_mobility_state[:141], time_lst[:141]), 7)
            print(state_name, first_wave)
            res.append(first_wave)

        df_mobility_first_wave = pd.DataFrame(res)
        df_mobility_first_wave["state"] = state_name_lst
        df_mobility_first_wave = df_mobility_first_wave[["state", "t0_time", "t0_index", "t0_val",
                                                        "t1_time", "t1_index", "t1_val",
                                                         "t2_time", "t2_index", "t2_val",
                                                         "t3_time", "t3_index", "t3_val",
                                                         "t4_time", "t4_index", "t4_val"]]
        # df_mobility_first_wave.to_csv(f"data/Ts_each_state/{mobility_type.split('/')[-1].replace('us_mobility_change_by_date_', '')}", index=False)
