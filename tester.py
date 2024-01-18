import pandas as pd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
pendulum_data = pd.read_excel('data2.xlsx', index_col=0)
chaos_pendulum_data = pd.read_excel('data.xlsx', index_col=0)
chaos_pendulum_array = chaos_pendulum_data.values
def excel_to_pandas(panda,x,y):
    x_s = [panda[i][x] for i in range(len(panda))]
    y_s = [panda[i][y] for i in range(len(panda))]
    x_and_y_data = [[],[]]
    shift = min(y_s)/2
    for i in range(len(x_s)):
        x_and_y_data[0].append(x_s[i]*100)
        x_and_y_data[1].append(y_s[i]*100-shift)


    return x_and_y_data
for i in range(4):
    data = excel_to_pandas(chaos_pendulum_array,3*i,3*i+1)
    data[0] = data[0][:250:3]
    data[1] = data[1][:250:3]

    if i == 0 and len(data[1])>277:
        data[1][278]  = -0.200
    t = [chaos_pendulum_array[i*3][2] for i in range(len(data[1]))]
    plt.scatter(data[0],data[1],c=t,cmap='Blues_r')
    plt.colorbar()
    plt.xlabel('x, см')
    plt.ylabel('y, см')
    plt.show()


