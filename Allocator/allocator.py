import pandas as pd
import numpy as np
import re
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

constraints = {
    'gpu': None,
    'wattage': 0,
    'socket': None,
    'supported_mem': None,
    'chipset': None,
    'form_factor': None,
    'max_memory': None,
    'memory_slots': None,
    'M.2': 0,
    'SATA': 0,
}

def fixnan(series):
    return series.where(pd.notnull(series), None)

def replace_nan_with_none(value):
    if pd.isna(value):
        return None
    return value


def re_weight(weights_dict, invert_key=None, epsilon=0.001):
    keys = list(weights_dict.keys())
    weights = np.array([weights_dict[key] for key in keys])
    
    if invert_key is not None:
        weights[keys.index(invert_key)] = 1 / (weights[keys.index(invert_key)] + epsilon)
    
    normalized_weights = weights / weights.sum()
    
    normalized_weights_dict = {keys[i]: normalized_weights[i] for i in range(len(keys))}
    
    return normalized_weights_dict
    

preference_preset = {
    'workstation': {
        'allocation': {'cpu': 0.3, 'mobo': 0.2, 'ram': 0.2, 'gpu': 0.3, 'sto': 0.1, 'psu': 0.1},
        'cpu': {'core_count': 0.5, 'boost_clock': 0.2, 'cache': 0.3, 'tdp': 0.1, 'passmark': 0.4, 'blender': 0.3, 'cinebench': 0.3},
        'motherboard': {'max_memory': 0.3, 'memory_slots': 0.2, 'SATA': 0.1, 'BAND': 0.1, 'PCIe': 0.2, 'OC': 0.1,},
        'memory': {'total_capacity_gb': 1, 'first_word_latency': 0.25, 'cas_latency': 0.25, 'speed_mhz': 0.4, 'price_per_gb':0.1, 'num_modules': 0.3},
        'gpu': {'memory': 4, 'memory_ver': 1, 'bus_width': 2, 'TDP': 0.1, 'g3dmark': 1.4, 'api_score': 1.2, 'blender_score': 1.2, 'fps_score': 1},
        'storage': {'capacity': 0.4, 'speed': 0.3, 'price_per_gb': 0.2, 'cache': 0.1},
        'psu': {'efficiency': 0.3, 'wattage': 0.6, 'modular': 0.1}
    },
    'gaming': {
        'allocation': {'cpu': 0.35, 'mobo': 0.3, 'ram': 0.1, 'gpu': 0.4, 'sto': 0.2, 'psu': 0.1},
        'cpu': {'core_count': 0.3, 'boost_clock': 0.8, 'cache': 0.5, 'tdp': 0.2, 'passmark': 0.4, 'blender': 0.3, 'cinebench': 0.3},
        'motherboard': {'max_memory': 0.3, 'memory_slots': 0.2, 'SATA': 0.1, 'BAND': 0.1, 'PCIe': 0.2, 'OC': 0.1,},
        'memory': {'total_capacity_gb': 1, 'first_word_latency': 0.25, 'cas_latency': 0.25, 'speed_mhz': 0.4, 'price_per_gb':0.1},
        'gpu': {'memory': 1, 'memory_ver': 1, 'bus_width': 1, 'TDP': 0.5, 'g3dmark': 1, 'api_score': 1, 'blender_score': 1, 'fps_score': 2},
        'storage': {'capacity': 0.3, 'speed': 0.5, 'price_per_gb': 0.1, 'cache': 0.1},
        'psu': {'efficiency': 0.3, 'wattage': 0.6, 'modular': 0.1}
    },
    'general': {
        'allocation': {'cpu': 0.3, 'mobo': 0.3, 'ram': 0.1, 'gpu': 0.3, 'sto': 0.2, 'psu': 0.1},
        'cpu': {'core_count': 0.5, 'boost_clock': 0.5, 'cache': 0.5, 'tdp': 0.3, 'passmark': 0.3, 'blender': 0.3, 'cinebench': 0.4},
        'motherboard': {'max_memory': 0.3, 'memory_slots': 0.2, 'SATA': 0.1, 'BAND': 0.1, 'PCIe': 0.2, 'OC': 0.1,},
        'memory': {'total_capacity_gb': 1, 'first_word_latency': 0.25, 'cas_latency': 0.25, 'speed_mhz': 0.4, 'price_per_gb':0.1},
        'gpu': {'memory': 2.2, 'memory_ver': 1, 'bus_width': 1, 'TDP': 0.5, 'g3dmark': 1.2, 'api_score': 1.2, 'blender_score': 1.2, 'fps_score': 1.2},
        'storage': {'capacity': 0.4, 'speed': 0.4, 'price_per_gb': 0.1, 'cache': 0.1},
        'psu': {'efficiency': 0.4, 'wattage': 0.5, 'modular': 0.1}
    },
    'server': {
        'allocation': {'cpu': 0.35, 'mobo': 0.3, 'ram': 0.2, 'gpu': 0.1, 'sto': 0.3, 'psu': 0.1},
        'cpu': {'core_count': 0.5, 'boost_clock': 0.2, 'cache': 0.3, 'tdp': 0.5, 'passmark': 0.4, 'blender': 0.3, 'cinebench': 0.3},
        'motherboard': {'max_memory': 0.3, 'memory_slots': 0.2, 'SATA': 0.1, 'BAND': 0.1, 'PCIe': 0.2, 'OC': 0.1,},
        'memory': {'total_capacity_gb': 1, 'first_word_latency': 0.1, 'cas_latency': 0.2, 'speed_mhz': 0.4, 'price_per_gb':0.3},
        'gpu': {'memory': 2.2, 'memory_ver': 1, 'bus_width': 1, 'TDP': 0.5, 'g3dmark': 1.2, 'api_score': 1.2, 'blender_score': 1.2, 'fps_score': 1.2},
        'storage': {'capacity': 0.4, 'speed': 0.4, 'price_per_gb': 0.1, 'cache': 0.1},
        'psu': {'efficiency': 0.5, 'wattage': 0.5, 'modular': 0}
    },
    'custom': {
        'allocation': {'cpu': 0.3, 'mobo': 0.2, 'ram': 0.1, 'gpu': 0.3, 'sto': 0.2, 'psu': 0.1},
        'cpu': {'core_count': 0.5, 'boost_clock': 0.5, 'cache': 0.5, 'tdp': 0.1, 'passmark': 0.3, 'blender': 0.3, 'cinebench': 0.4},
        'motherboard': {'max_memory': 0.3, 'memory_slots': 0.2, 'SATA': 0.1, 'BAND': 0.1, 'PCIe': 0.2, 'OC': 0.1,},
        'memory': {'total_capacity_gb': 1, 'first_word_latency': 0.25, 'cas_latency': 0.25, 'speed_mhz': 0.4, 'price_per_gb':0.1},
        'gpu': {'memory': 2.2, 'memory_ver': 1, 'bus_width': 1, 'TDP': 0.5, 'g3dmark': 1.2, 'api_score': 1.2, 'blender_score': 1.2, 'fps_score': 1.2},
        'storage': {'capacity': 0.4, 'speed': 0.4, 'price_per_gb': 0.1, 'cache': 0.1},
        'psu': {'efficiency': 0.4, 'wattage': 0.5, 'modular': 0.1}
    },
}


def select_cpu(preference, budget, explored, balance=0.5):
    cpu_pref = preference['cpu']
    normalized_pref = re_weight(cpu_pref)
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/CPU/cpu_final.csv')
    df = pd.read_csv(csv_path)
    
    #ignore any mismatch cpus
    df = df.drop(explored)

    # find result from preferences
    df['performance'] = sum(df[col] * weight for col, weight in normalized_pref.items())
    df['power-perf'] = df['performance'] / df['tdp']
    df['price-perf'] = (df['performance'] / df['price']) if df['price'] is not None else 0
    df['result'] = (df['power-perf']*balance + df['price-perf']*(1 - balance))
    
    df = df.sort_values(by='result', ascending=False)
        
    for index, row in df.iterrows():
        if row['price'] == '':
            continue
        part_price = float(row['price'])
        if part_price <= budget:
            part = row
            break

    # CLEAN NAN VALUES BEFORE SUBMITTING
    part = fixnan(part)
    
    constraints['gpu'] = part['graphics'] if part['graphics'] != '' else None
    constraints['wattage'] = part['tdp_spec'] if part['tdp_spec'] != '' else 0
    constraints['socket'] = str(part['socket']) if part['socket'] != '' else None
    constraints['supported_mem'] = part['supported_mem'] if part['supported_mem'] != '' else None
    constraints['chipset'] = str(part['supported_chipset']) if part['supported_chipset'] != '' else None
    
    leftover = budget - part['price']
    component = {
        'type': 'CPU',
        'index': part['index'],
        'name': part['name'],
        'price': part['price'], 
        'description': {
            'Speed' : f"{part['core_count_spec']} Cores @ {part['boost_clock_spec']}GHz",
            'Hyper-Threading': part['smt'],
            'Cache': part['cache_spec'],
            'Internal GPU': part['graphics'] if part['graphics'] != '' else None,
            'Power Consumption': f"{part['tdp_spec']}",
        },
        'score': round(part['result']*100, 3)
    }
    
    return component, leftover


def select_motherboard(preference, mb_budget, include_wifi=False):
    pref = preference['motherboard']
    normalized_pref = re_weight(pref)
    #print(f'motherboard preference: {pref}')
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/Motherboard/motherboard_final.csv')
    df = pd.read_csv(csv_path)
    
    #filter down from preferences
    socket = constraints['socket']
    chipset = constraints['chipset']
    
    if socket != 'nan':
        df = df[df['socket'] == socket]
    if chipset != 'nan':
        df = df[df['chipset'].isin(chipset.split(', '))]
    if include_wifi:
        df = df[df['WIFI'] == True]
    
    #print("LENGTH OF DF: ", len(df))
    if len(df) < 1:
        print('unable to fit desired motherboard with bound parameters')
        return None, mb_budget
    
    # find result from preferences
    df['result'] = sum(df[col] * weight for col, weight in normalized_pref.items())
    df = df.sort_values(by='result', ascending=False)
    
    #begin looking for parts
    part = None
    for index, row in df.iterrows():
        if row['price'] == '':
            continue
        part_price = float(row['price'])
        if part_price <= mb_budget:
            part = row
            break
    
    if part is None:
        df = df.sort_values(by='price')
        cheapest = df.iloc[0]
        print(f"No Motherboard within provided budget found. Cheapest found part is {cheapest['price']}, budget {mb_budget}")
        return None, mb_budget
    
    leftover = mb_budget - part['price']
    # CLEAN NAN VALUES BEFORE SUBMITTING
    part = fixnan(part)
    
    # re-calculate constraints
    constraints['form_factor'] = part['form_factor']
    constraints['max_memory'] = part['max_memory_spec']
    constraints['memory_slots'] = part['memory_slots_spec']
    constraints['M.2'] = part['M.2'] if part['M.2'] != '' else None
    constraints['SATA'] = part['SATA_spec'] if part['SATA_spec'] != '' else None
    
    component = {
        'type': 'Motherboard',
        'name': part['name'],
        'price': part['price'], 
        'description': {
            'Chipset' : part['chipset'] if part['chipset'] else None,
            'Memory Slots': part['memory_slots_spec'],
            'Socket': part['socket'],
            'Form Factor': part['form_factor'],
            'Overclocking': part['OC_spec']
        },
        'score': round(part['result']*100, 3)
        }
    
    return component, leftover


def select_memory(preference, budget):
    pref = preference['memory']
    normalized_pref = re_weight(pref)
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/RAM/memory_final.csv')
    df = pd.read_csv(csv_path)
    
    #filter down from preferences
    supported_DDR = constraints['supported_mem']
    #print(supported_DDR) #prints DDR5
    
    supported_DDR = re.findall(r'\d+', supported_DDR)
    max_memory = constraints['max_memory']
    no_slots = constraints['memory_slots']
    
    df = df[df['DDR_version'].isin([float(version) for version in supported_DDR])]
    if max_memory is not None:
        df = df[df['capacity'] <= max_memory]
    if no_slots is not None:
        df = df[df['num_modules'] <= no_slots]

    if len(df) < 1:
        print('unable to fit desired RAM with bound parameters')
        return None, budget
    
    # find result from preferences
    df['result'] = sum(df[col] * weight for col, weight in normalized_pref.items())
    df = df.sort_values(by='result', ascending=False)
    
    #begin looking for parts
    part = None
    for index, row in df.iterrows():
        if row['price'] == '':
            continue
        part_price = float(row['price'])
        if part_price <= budget:
            part = row
            break
    
    if part is None:
        df = df.sort_values(by='price')
        cheapest = df.iloc[0]
        print(f"No Memory within provided budget found. Cheapest found part is {cheapest['name']} at {cheapest['price']}, budget {budget}")
        return None, budget
    
    # CLEAN NAN VALUES BEFORE SUBMITTING
    part = fixnan(part)
    leftover = budget - part['price']
    
    component = {
        'type': 'Memory',
        'name': part['name'],
        'price': part['price'], 
        'description': {
            'Total Capacity': part['capacity'],
            'Number of Modules' : f"{part['num_modules']} x {part['gb_per_module']}",
            'Memory Speed': part['speed'],
        },
        'score': round(part['result']*100, 3)
        }
    
    return component, leftover


def select_gpu(preference, budget):
    gpu_pref = preference['gpu']
    normalized_pref = re_weight(gpu_pref)
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/GPU/gpu_final.csv')
    df = pd.read_csv(csv_path)
    
    #ignore any mismatch cpus
    #df = df.drop(explored)

    # find result from preferences
    df['performance'] = sum(df[col] * weight for col, weight in normalized_pref.items())
    
    #df['power-perf'] = df['performance'] / df['TDP']
    #df['price-perf'] = (df['performance'] / df['price']) if df['price'] is not None else 0
    #df['result'] = (df['power-perf']*balance + df['price-perf']*(1 - balance))
    df['result'] = df['performance']
    
    df = df.sort_values(by='result', ascending=False)
        
    for index, row in df.iterrows():
        if row['price'] == '':
            continue
        part_price = float(row['price'])
        if part_price <= budget:
            part = row
            break

    # CLEAN NAN VALUES BEFORE SUBMITTING
    part = fixnan(part)
    leftover = budget - part['price']
    constraints['wattage'] += part['TDP_spec']
    
    component = {
        'type': 'GPU',
        'name': part['chipset'],
        'price': part['price'], 
        'description': {
            'vendor': part['name'],
            'V-RAM': f"{part['VRAM']} GB",
            'Bus Width': f"{part['bus_width_spec']}bit",
            'Power Consumption': f"{part['TDP_spec']}W",
        },
        'score': round(part['result']*100, 3)
    }
    
    return component, leftover


def select_psu(preference, budget):
    pref = preference['psu']
    normalized_pref = re_weight(pref)
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/PSU/psu_final.csv')
    df = pd.read_csv(csv_path)
    
    #filter out PSUs with lower wattage than required
    pc_power_consumption = constraints['wattage']
    df = df[df['Watts'] >= 2 * pc_power_consumption]
    
    #sort remaining PSUs on performance
    df['result'] = sum(df[col] * weight for col, weight in normalized_pref.items())
    df = df.sort_values(by='result', ascending=False)
    
    part = None
    for index, row in df.iterrows():
        if row['price'] == '':
            continue
        part_price = float(row['price'])
        if part_price <= budget:
            part = row
            break
        
    if part is None:
        df = df.sort_values(by='price')
        cheapest = df.iloc[0]
        print(f"No PSU within provided budget found. Cheapest found part is {cheapest['price']}, budget {budget}")
        return None, budget
        
        
    # CLEAN NAN VALUES BEFORE SUBMITTING
    part = fixnan(part)
    leftover = budget - part['price']
    
    component = {
        'type': 'PSU',
        'name': part['name'],
        'price': part['price'], 
        'description': {
            'Wattage': part['Watts'],
            'Efficiency': f"80+ {part['efficiency_type']}",
            'Modular': part['modularity'],
            'Type': part['type'],
        },
        'score': round(part['result']*100, 3)
    }
    
    return component, leftover


def select_storage(preference, budget):
    pref = preference['storage']
    normalized_pref = re_weight(pref)
    
    #load csv file
    csv_path = os.path.join(BASE_DIR, '../Datasheets/Storage/storage_final.csv')
    df = pd.read_csv(csv_path)
    
    # dont need any unknown price rows
    df.dropna(subset=['price'], inplace=True)
    
    leftover = budget
    storage_devices = []
    while (constraints['SATA'] > 1) or (constraints['M.2'] > 1):
        #sort remaining PSUs on performance
        df['result'] = sum(df[col] * weight for col, weight in normalized_pref.items())
        df = df.sort_values(by='result', ascending=False)
    
        part = None
        for index, row in df.iterrows():
            if row['price'] == '':
                continue
            part_price = float(row['price'])
            if part_price <= leftover:
                part = row
                break
        
        if part is None:
            df = df.sort_values(by='price')
            cheapest = df.iloc[0]
            print(f"No more Storage device(s) within provided budget found. Cheapest found part is {cheapest['price']}, budget {leftover}")
            return storage_devices, leftover
            
        # CLEAN NAN VALUES BEFORE SUBMITTING
        part = fixnan(part)
        component = {
            'type': part['type'],
            'name': part['name'],
            'price': part['price'], 
            'description': {
                'Capacity': f"{part['capacity_spec'] * 1000}GB",
                'Interface': part['interface'],
                'Cache': f"{part['cache_spec']}MB",
            },
            'score': round(part['result']*100, 3)
        }
        
        if part['form_factor'] == 'M.2':
            constraints['M.2'] -= 1
        else:
            constraints['SATA'] -= 1
        
        leftover = leftover - part['price'] 
        
        storage_devices.append((component)) 
        #print(storage_devices) 
        
    return storage_devices, leftover


def allocator(total_budget, preference):
    #preference = preference_preset[preference]
    # HIGHER BALANCE = power-performance, LOWER BALANCE = price-performance
    balance = 0.8
    
    #total_budget = 700
    allocation = re_weight(preference['allocation'])

    budget = {}
    for item in allocation:
        budget[item] = allocation[item] * total_budget
        print(f"Budget: {budget[item]} for {item}")
    
    cpu_budget = total_budget * allocation['cpu']
    mb_budget = total_budget * allocation['mobo']
    ram_budget = total_budget * allocation['ram']
    gpu_budget = total_budget * allocation['gpu']
    sto_budget = total_budget * allocation['sto']
    psu_budget = total_budget * allocation['psu']
    
    explored_indices = {'cpu': [], 'mobo': [], 'ram': []}
    while True:
        leftover = 0
        
        if cpu_budget > 5:
            cpu, leftover = select_cpu(preference, cpu_budget, explored_indices['cpu'], balance)
            explored_indices['cpu'].append(cpu['index'] - 1) # -1 for true index
        else:
            cpu = None
        
        
        if mb_budget > 5:
            mb, leftover = select_motherboard(preference, mb_budget + leftover, include_wifi=False)
            if mb is None:
                print("[DEBUG] retrying...")
                continue
            else:
                if explored_indices['cpu'] != []:
                    explored_indices['cpu'].pop()
        else:
            mb = None
        
        if gpu_budget > 5:
            gpu, leftover = select_gpu(preference, gpu_budget + leftover)
        else:
            gpu = None
        
        if ram_budget > 5:
            ram, leftover = select_memory(preference, ram_budget + leftover)
        else:
            ram = None

        if sto_budget > 5:
            storage_devices, leftover = select_storage(preference, sto_budget + leftover)
        else:
            storage_devices = None
        
        if psu_budget > 5:
            psu, leftover = select_psu(preference, psu_budget + leftover)
        else:
            psu = None
 
        break
    
    part_list = {'cpu':cpu, 
                 'motherboard':mb, 
                 'ram':ram, 
                 'gpu':gpu, 
                 'storage': storage_devices, 
                 'psu':psu}
    
    return part_list, constraints, round(leftover, 2)
