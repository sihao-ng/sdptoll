# Introduced CSV lists

import time
import re
import csv
import json
import math
import numpy
from gurobipy import *

master_list_demand = []     # tuple: (customer, sku, demand, value_of_shipment(for GST calculation purpose))
master_list_product = []    # tuple: (sku, hs_code, weight/unit, height, width, length, iced, $/unit, origin_country)
master_list_tax = []        # tuple: (origin, destination, 1-gst/0-tariff, hs_code, type, rate)

master_list_transport = []  # tuple: (origin, destination, $/kg)
master_list_warehouse = []  # tuple: (location, fixed cost, $/m3)

I = []    # Hub country names
J = []    # Customer country names
L = []    # Plant country names

N_I = 0   # Number of Hubs
N_J = 0   # Number of Customers
N_K = 0   # Number of Products
N_L = 0   # Number of Plants






def main():
    load_master_list_demand()           # Load lookup tables
    load_master_list_product()
    load_master_list_tax()
    load_master_list_locations()

    method = -1                         # Choose Solving Method Algorithm (-1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex)
    run_solver(method)






# Solver Function
def run_solver(method):
    model = Model('SDP Group 10')       # Initialize Model
    model.setParam('Method', method)    # Specify Solving Method

    # Decision Variables
    Y = numpy.empty((N_I + 1, N_J + 1, N_K + 1), dtype = object)   # Indicator variable if Warehouse i serves Customer j Product k
    Z = numpy.empty((N_L + 1, N_I + 1, N_K + 1), dtype = object)   # Amount of Product k flowing from Plant l to Warehouse i
    X = numpy.empty((N_I + 1), dtype = object)

    for i in range(1, N_I + 1):                                 # 1, 2, 3, ... I
        for j in range(1, N_J + 1):
            for k in range(1, N_K + 1):
                Y[i, j, k] = model.addVar(vtype=GRB.BINARY, name='Y{}.{}.{}'.format(str(i), str(j), str(k)))

    for l in range(1, N_L + 1):
        for i in range(1, N_I + 1):
            for k in range(1, N_K + 1):
                Z[l, i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Z{}.{}.{}'.format(str(l), str(i), str(k)))

    for i in range(1, N_I + 1):
        X[i] = model.addVar(lb=0, vtype=GRB.BINARY, name='X{}'.format(str(i)))

    model.update()

    # Objective Function
    model.setObjective(
        quicksum(
            quicksum(
                quicksum(
                    (
                        (
                            transWC(i, j, k)
                            +
                            whVar(i, k)
                            +
                            unit_price(k) * (GST(i, j, k) * W(i, j, k) + tar_hub_to_cus(i, j, k))
                        ) * d(j, k) * Y[i, j, k]
                    ) for k in range(1, N_K + 1)
                ) for j in range(1, N_J + 1)
            ) for i in range(1, N_I + 1)
        )
        +
        quicksum(
            quicksum(
                quicksum(
                    (
                        (
                            transPW(l, i, k)
                            +
                            unit_price(k) * tar_plt_to_hub(l, i, k)
                        ) * Z[l, i, k]
                    ) for k in range(1, N_K + 1)
                ) for i in range(1, N_I + 1)
            ) for l in range(1, N_L + 1)
        )
        +
        quicksum(
            whFix(i) * X(i) for i in range(1, N_I + 1)
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

    # c2 (N/A - No max number of facilities)
    # c3 (N/A - No min number of facilities)
    # c4 (N/A - No diff warehouse types)
    # c5 (N/A - No warehouse capacity limit)

    # c6
    for i in range(1, N_I + 1):
        for k in range(1, N_K + 1):
            model.addConstr(
                quicksum(Z[l, i, k] for l in range(1, N_L + 1)) == quicksum((d(j, k) * Y[i, j, k]) for j in range(1, N_J + 1))
                , name = 'c6_I{}K{}'.format(str(i), str(k))
            )

    # c7 (N/A - No plant capacity limit)

    # c8
    for i in range(1, N_I + 1):
        for j in range(1, N_J + 1):
            for k in range(1, N_K + 1):
                model.addConstr(Y[i, j, k] <= X[i], name = 'c8_I{}J{}K{}'.format(str(i), str(j), str(k)))

    # c9 (N/A - Y already initialized as binary variable)
    # c10 (N/A - X already initialized as binary variable)
    # c11 (N/A - Z already initialized as continuous variable with lower bound 0)

    model.optimize()        # Solve Model
    model.printAttr('x')    # Print Results

    #model.write('fyp.lp')  # Save Model Details
    #model.write('fyp.mps')






# Demand Function (Returns number of units)
def d(j, k):
    sku = master_list_product[k][0]
    for tuple in master_list_demand:
        if tuple[0] == j and tuple[1] == sku:
            return tuple[2]
    return 0

# GST Function (Returns x% GST)
def GST(i, j, k):
    origin_country = I[i]
    destination_country = J[j]
    hs_code = master_list_product[k][1]

    for tuple in master_list_tax:
#        if tuple[0] == origin_country and tuple[1] == destination_country and tuple[2] and tuple[3] == hs_code:
        if tuple[0] == origin_country and tuple[1] == destination_country and tuple[2] and match(hs_code, tuple[3]):
            if tuple[4] == 0:           # Type is ad valorem
                return tuple[5]
            else:
                return 0.0
    return 0.0

# Tariff Functions
def tar(origin, destination, hs_code):
    for tuple in master_list_tax:
        if tuple[0] == origin and tuple[1] == destination and not tuple[2] and match(hs_code, tuple[3]):
            if tuple[4] == 0:           # Type is ad valorem
                return tuple[5]
            else:
                print('{}'s tar could not be calculated! Will assume no tariff.')
                return 0.0
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

# Transport Functions
def trans(origin, destination, k):
    weight_per_unit = master_list_product[k][2]
    for tuple in master_list_transport:
        if tuple[0] == origin and tuple[1] == destination:
            return tuple[2] * weight_per_unit
    return 0

def transWC(i, j, k):
    origin = I[i]
    destination = J[j]
    return trans(origin, destination, k)

def transPW(l, i, k):
    origin = L[l]
    destination = I[i]
    return trans(origin, destination, k):

# Unit Price Function (Returns $/unit)
def unit_price(k):
    return master_list_product[k][7]

# GST Eligibility Function (Returns 0 or 1)
def W(i, j, k):
    value_of_shipment = d(j, k) * unit_price(k)
    customer_country = J[j]

    if customer_country == 'singapore' and value_of_shipment <= 400:
        return 0
    return 1

# Warehouse Cost Functions
def whVar(i, k):
    location = I[i]
    volume_per_unit = master_list_product[k][3] * master_list_product[k][4] * master_list_product[k][5]
    #weight_per_unit = master_list_product[k][2]
    for tuple in master_list_warehouse:
        if tuple[0] == location:
            return tuple[2] * volume_per_unit
            #return tuple[2] * weight_per_unit
    return 0.0

def whFix(i):
    location = I[i]
    for tuple in master_list_warehouse:
        if tuple[0] == location:
            return tuple[1]
    retutn 0.0






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

        for row in reader:          # Add country names to respective lists
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

def load_master_list_transport():
    global master_list_transport

    f = open('master_list_transport.csv', newline = '')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        origin = row[0]
        destination = row[1]
        unit_price = float(row[2])
        master_list_tax.append((origin,destination,unit_price))
    f.close()

def load_master_list_warehouse():
    global master_list_warehouse

    f = open('master_list_warehouse.csv', newline = '')
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        location = row[0]
        fixed_cost = float(row[1])
        price_per_m3 = float(row[2])
        master_list_tax.append((location, fixed_cost, price_per_m3))

        #price_per_kg = float(row[2])
        #master_list_tax.append((location, fixed_cost, price_per_kg))
    f.close()






def match(string, wildcard):
    if wildcard = '-':      # Will always match
        return 1

    regex = re.compile(wildcard)
    if re.match(regex, string):
        return 1
    return 0






if __name__ == '__main__':
    main()