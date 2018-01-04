# Introduced CSV lists

import time
import csv
import json
import math
import numpy
from gurobipy import *

master_list_demand = []         # customer, sku, demand, value_of_shipment
master_list_product = []        # sku, hs_code, weight/unit, height, width, length, iced, $/unit, origin_country
master_list_tax = []            # origin, destination, 1-gst/0-tariff, hs_code, type, rate

I = []    # Hubs
J = []    # Customers
L = []    # Plants

N_I = 0   # Number of Hubs
N_J = 0   # Number of Customers
N_K = 0   # Number of Products
N_L = 0   # Number of Plants

def main():
    # Load lookup tables
    load_master_list_demand()
    load_master_list_product()
    load_master_list_tax()
    load_master_list_locations()

    # Choose Solving Method Algorithm (-1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex)
    method = -1
    run_solver(method)

# Demand Function (Returns number of units)
def d(j, k):
    sku = master_list_product[k][0]
    for tuple in master_list_demand:
        if tuple[0] == j and tuple[1] == sku:
            return tuple[2]
    return 0

# Unit Price Function (Returns $/unit)
def unit_price(k):
    return master_list_product[k][7]

# GST Function (Returns x% GST)
def GST(i, j, k):
    origin_country = I[i]
    destination_country = J[j]
    hs_code = master_list_product[k][1]

    for tuple in master_list_tax:
        if tuple[0] == origin_country and tuple[1] == destination_country and tuple[2] and tuple[3] == hs_code:
            if tuple[4] == 0:
                return tuple[5]
            else:
                return 0.0
    return 0.0

# GST Eligibility Function (Returns 0 or 1)
def W(i, j, k):
    value_of_shipment = d(j, k) * unit_price(k)
    customer_country = J[j]

    if customer_country == 'singapore' and value_of_shipment <= 400:
        return 0
    return 1

# Tariff Functions
def tar(origin, destination, hs_code):
    for tuple in master_list_tax:
        if tuple[0] == origin and tuple[1] == destination and not tuple[2] and tuple[3] == hs_code:
            return tuple[5]
    return 0.0

def tar_plt_to_hub(l, i, k):
    origin = L[l]
    destination = I[i]
    hs_code = master_list_product[k][1]
    return tar(origin, destination, hs_code)

def tar_hub_to_cus(i, j, k):
    origin = I[i]
    destination = J[j]
    hs_code = master_list_product[k][1]
    return tar(origin, destination, hs_code)


# Solver Function
def run_solver(method):
    # Initialize Model
    model = Model('SDP Group 10')

    # Specify Solving Method
    model.setParam('Method', method)

    # Decision Variables
    Y = numpy.empty((N_I + 1, N_J + 1, N_K + 1), dtype = object)   # Indicator variable if Warehouse i serves Customer j Product k
    Z = numpy.empty((N_L + 1, N_I + 1, N_K + 1), dtype = object)   # Amount of Product k flowing from Plant l to Warehouse i

    for i in range(1, N_I + 1):                                 # 1, 2, 3, ... I
        for j in range(1, N_J + 1):
            for k in range(1, N_K + 1):
                Y[i, j, k] = model.addVar(vtype=GRB.BINARY, name='Y{}.{}.{}'.format(str(i), str(j), str(k)))

    for l in range(1, N_L + 1):
        for i in range(1, N_I + 1):
            for k in range(1, N_K + 1):
                Z[l, i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Z{}.{}.{}'.format(str(l), str(i), str(k)))

    model.update()

    # Objective Function
    model.setObjective(
        quicksum(
            quicksum(
                quicksum(
                    (unit_price(k) * (GST(i, j, k) * W(i, j, k) + tar_hub_to_cus(i, j, k)) * d(j, k) * Y[i, j, k]) for k in range(1, N_K + 1)
                ) for j in range(1, N_J + 1)
            ) for i in range(1, N_I + 1)
        )
        +
        quicksum(
            quicksum(
                quicksum(
                    (unit_price(k) * tar_plt_to_hub(l, i, k) * Z[l, i, k]) for k in range(1, N_K + 1)
                ) for i in range(1, N_I + 1)
            ) for l in range(1, N_L + 1)
        )
        , GRB.MINIMIZE
    )

    # Constraints
    # c1
    for j in range(1, N_J + 1):
        for k in range(1, N_K + 1):
            model.addConstr(
                quicksum(Y[i, j, k] for i in range(1, N_I + 1)) == 1
                , name = 'c1_J{}_K{}'.format(str(j), str(k))
            )

    # c2 (N/A - No warehouse fixed cost)
    # c3 (N/A - No warehouse fixed cost)
    # c4 (N/A - No warehouse fixed cost)
    # c5 (N/A - No warehouse capacity limit)
    # c6
    for i in range(1, N_I + 1):
        for k in range(1, N_K + 1):
            model.addConstr(
                quicksum(Z[l, i, k] for l in range(1, N_L + 1)) == quicksum((d(j, k) * Y[i, j, k]) for j in range(1, N_J + 1))
                , name = 'c6_I{}K{}'.format(str(i), str(k))
            )

    # c7 (N/A - No plant capacity limit)
    # c8 (N/A - No warehouse fixed cost)
    # c9 (N/A - Y already initialized as binary variable)
    # c10 (N/A - No X variable)
    # c11 (N/A - Z already initialized as continuous variable with lower bound 0)

    # Solve Model
    model.optimize()

    # Print Results
    model.printAttr('x')

    # Save Model Details
    model.write('fyp.lp')
    model.write('fyp.mps')

def load_master_list_demand():
    global master_list_demand

    f = open('master_list_demand.csv', newline = '')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        customer = int(row[0])
        sku = row[1]
        demand = int(row[2])
        master_list_demand.append((customer, sku, demand))
    f.close()

def load_master_list_product():
    global master_list_product
    global N_K

    f = open('master_list_product.csv', newline = '')
    reader = csv.reader(f)
    header = next(reader)

    master_list_product.append((0,0,0,0,0,0,0,0,0))

    for row in reader:
        sku = row[0]
        hs_code = row[1]
        unit_weight = float(row[2])
        height = float(row[3])
        width = float(row[4])
        length = float(row[5])
        iced = int(row[6])
        unit_price = float(row[7])
        origin_country = row[8]
        master_list_product.append((sku,hs_code,unit_weight,height,width,length,iced,unit_price,origin_country))
    f.close()
    N_K = len(master_list_product) - 1

def load_master_list_tax():
    global master_list_tax

    f = open('master_list_tax.csv', newline = '')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        origin = row[0]
        destination = row[1]
        gst_or_tariff = int(row[2])
        hs_code = row[3]
        type = int(row[4])
        rate = float(row[5])
        master_list_tax.append((origin,destination,gst_or_tariff,hs_code,type,rate))
    f.close()

def load_master_list_locations():
    global I
    global J
    global L
    global N_I
    global N_J
    global N_L

    I.append(0)     # First element is empty
    J.append(0)     # First element is empty
    L.append(0)     # First element is empty

    for i, filename in enumerate(['master_list_location_hub.csv', 'master_list_location_customer.csv', 'master_list_location_plant.csv']):
        f = open(filename, newline = '')
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if i == 0:
                I.append(row[1])
            elif i == 1:
                J.append(row[1])
            elif i == 2:
                L.append(row[1])
    f.close()
    N_I = len(I) - 1
    N_J = len(J) - 1
    N_L = len(L) - 1

if __name__ == '__main__':
    main()