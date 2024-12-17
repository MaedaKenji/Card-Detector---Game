import math

def f(v,i):
    return float(v/i)

def process_input_text(input_text):
    if len(input_text) < 7:
        return 'Input harus berupa string dengan minimal 7 karakter.'

    result = []
    for i in range(0, len(input_text),4):
        base_part = input_text[i]
        decimal_part = input_text[i+1:i+4]
        if len(base_part) == 1 and len(decimal_part) == 3:
            result.append(round(float(base_part[0] + '.' + decimal_part), 3))
    return result

def process_input_text2(input_text):
    result = []
    for i in range(0, len(input_text),3):
        base_part = input_text[i]
        decimal_part = input_text[i+1:i+3]
        if len(base_part) == 1 and len(decimal_part) == 2:
            result.append(round(float(base_part[0] + '.' + decimal_part), 3))
    return result

input_text_va = "0267029903470397042704190383033802970261"
va = process_input_text(input_text_va)
print(f"VA = {va}")
input_text_it = "4550452044804430443044804560463046704700" 
it = process_input_text(input_text_it)
print(f"IT = {it}")
input_text_ic = "0365144024503630474055005840593058805790"
ic = process_input_text(input_text_ic)
input_text_il = "4300445046004660444039303300270022201830"
il = process_input_text(input_text_il)
print(f"IC = {ic}")
print(f"IL = {il}")
iir = "0240026003100360039003900350030002600220"
ir = process_input_text(iir)
print(f"IR = {ir}")
z = []
xc = []
xl = []

for a in range(len(va)):
    xc.append(round(va[a]/ic[a]*1000,4))
print(f"XC = {xc}")

for a in range(len(va)):
    xl.append(round4(va[a]/il[a]*1000,4))
print(f"XL = {xl}")


for a in range(len(va)):
    z.append(round(va[a]/it[a]*1000,4))
print(f"Z = {z}")

theta = []
for a in range(len(ir)):
    arctan = math.atan((ic[a]-il[a])/ir[a])
    # theta.append(round(arctan,4))
    theta.append(round(math.degrees(arctan),4))

print(f"THETA = {theta}")


# import numpy as np
# import matplotlib.pyplot as plt

# # Data THETA dalam derajat
# # THETA = [-86.5098, -85.0631, -81.7953, -70.7347, 37.5686, 76.0497, 82.1543, 84.6936, 85.9366, 86.8202,89,-89]
# THETA = [-86.5098, 170]

# # Buat plot untuk masing-masing THETA
# plt.figure(figsize=(6,6))
# plt.axhline(0, color='black',linewidth=1)
# plt.axvline(0, color='black',linewidth=1)

# for theta in THETA:
#     sin_theta = np.sin(np.radians(theta))
#     x_values = np.linspace(-10, 10, 100)
#     y_values = x_values * sin_theta
#     plt.plot(x_values, y_values, label=f'sin({theta}°)')

# # Labels dan title
# plt.title('Diagram Kartesius: f(x) = x * sin(THETA)')
# plt.xlabel('x')
# plt.ylabel('f(x) = x * sin(THETA)')
# plt.grid(True)
# plt.legend()

# # Show the plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Data theta in degrees
# THETA = [-86.5098, -85.0631, -81.7953, -70.7347, 37.5686, 76.0497, 82.1543, 84.6936, 85.9366, 86.8202]

# # Convert theta to radians for plotting on Cartesian plane
# theta_rad = np.deg2rad(THETA)

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(6, 6))

# # Set axis limits for a symmetric Cartesian plane
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])

# # Add Cartesian lines (x and y axis)
# ax.axhline(0, color='black',linewidth=0.5)
# ax.axvline(0, color='black',linewidth=0.5)

# # Plot theta points on Cartesian plane
# for angle in THETA:
#     x = np.cos(np.deg2rad(angle))
#     y = np.sin(np.deg2rad(angle))
#     ax.plot([0, x], [0, y], marker='o', label=f'{angle:.2f}°')
#     ax.text(x * 1.1, y * 1.1, f'{angle:.2f}°', fontsize=8)

# # Set grid and title
# ax.grid(True)
# ax.set_title('Cartesian Plot of THETA Values in Degrees')

# # Show plot
# plt.show()









# ivs = "0210031206050775"
# vs = process_input_text(ivs)
# print(f"VS = {vs}")
# vp = [2,4,6,7]

# for a in range(len(vp)):
#     print(vp[a]/vs[a])
    
# iip = "0950170023402640"
# ip = process_input_text(iip)
# print(f"IP = {ip}")
# iis = "0200045006300730"
# is_ = process_input_text(iis)
# print(f"IS = {is_}")

# for a in range(len(ip)):
#     print(ip[a]/is_[a])










# xl = []
# for a in range(len(vl)):
#     xl.append(round(vl[a]/(i[a]/1000),4))

# print(f"XL = {xl}")

# R = []
# for a in range(len(vr)):
#     R.append(round(vr[a]/(i[a]/1000),4))

# print(f"R = {R}")

# Z = []
# for a in range(len(vc)):
#     Z.append(round(math.sqrt((R[a]**2)+(xl[a]-xc[a])**2),4))

# print(f"Z = {Z}")

import numpy as np

# Data from the image (adjust as needed)
resistance_values = [100, 220, 330, 470, 680, 1000]
current_values = [17.5, 14.48, 12.65, 10.66, 8.7, 6.681]
current_values = np.array(current_values) / 1000
# Calculate power

power_values = np.array(current_values)**2 * np.array(resistance_values)



print("Resistance:", resistance_values)
print("Current:", current_values)
print("Power:", power_values)

i = [4.47, -1.63, 6.12]
i2 = [9.31, 6.41, 2.9]
i3 = [-4.8, -8.03, 3.2]

i = [4.47, -1.63, 6.12]
i2 = [9.31, 6.41, 2.9]
i3 = [-4.8, -8.03, 3.2]

# Panjang dari list (asumsikan semua list memiliki panjang yang sama)
panjang_list = len(i)

# Iterasi setiap elemen
for indeks in range(panjang_list):
    selisih = i[indeks] - (i2[indeks] + i3[indeks])
    selisih_mutlak = abs(selisih)
    print(f"Indeks ke-{indeks}: Selisih = {selisih}, Selisih Mutlak = {selisih_mutlak}")
    
import numpy as np

i2 = [9.31, 6.41, 2.9]
i3 = [-4.8, -8.03, 3.2]

result = np.array(i2) + np.array(i3)
print(result)


v1 = [1,3,5]
v2 = [1,3,5]
v3 = [1,3,5]

def process_input_text5(input_text):
    if len(input_text) < 7:
        return 'Input harus berupa string dengan minimal 7 karakter.'

    result = []
    for i in range(0, len(input_text),5):
        base_part = input_text[i]
        decimal_part = input_text[i+1:i+5]
        # print(base_part,decimal_part)
        if len(base_part) == 1 and len(decimal_part) == 4:
            result.append(round(float(base_part[0] + '.' + decimal_part), 4))
    return result




i1="006250189103149"
i2="020960629010490"
i3="029930899015300"

i1 = process_input_text5(i1)
i2 = process_input_text5(i2)
i3 = process_input_text5(i3)

i1=np.array(i1)
i2=np.array(i2)
i3=np.array(i3)

i1 = i1/1000
i2 = i2/1000
i3 = i3/1000

v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)

# for a in range(len(i1)):
#     print("resistance",v1[a]/(i1[a]))
    
# for a in range(len(i2)):
#     print("resistance",v2[a]/(i2[a]))
    
# for a in range(len(i3)):
#     print("resistance",v3[a]/(i3[a]))

def process_input_text2(input_text):
    result = []
    for i in range(0, len(input_text),4):
        base_part = input_text[i]
        decimal_part = input_text[i+1:i+3]
        print(base_part,decimal_part)
        if len(base_part) == 1 and len(decimal_part) == 2:
            result.append(round(float(base_part[0] + '.' + decimal_part), 3))
    return result

i4="326095201529"
i5="219063501041"
i6="126037106100"

i4 = process_input_text2(i4)
i5 = process_input_text2(i5)
i6 = process_input_text2(i6)

i4[2]= 15.29
i5[2]= 10.41

i4=np.array(i4)
i5=np.array(i5)
i6=np.array(i6)

i4 = i4/1000
i5 = i5/1000
i6 = i6/1000

v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)

print(i4)
print(i5)
print(i6)

for a in range(len(i4)):
    print("resistance",v1[a]/(i4[a]))
    
for a in range(len(i5)):
    print("resistance",v2[a]/(i5[a]))
    
for a in range(len(i6)):
    print("resistance",v3[a]/(i6[a]))