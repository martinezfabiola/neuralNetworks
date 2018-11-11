"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Description : question 2.

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

from neuralnetworkbp import *
from generador import *
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_point(tuple_array):

    x = []
    y = []
    for i in range(len(tuple_array)):
        x.append(tuple_array[i][0])
        y.append(tuple_array[i][1])

    return x,y


def plot_circle_points(x_points,y_points,radio,marca):
    m = '+'
    c = '#9D979E'

    if marca:
        m = '*'
        c = '#C847E3'

    fig = plt.figure(1)
    plt.axis([0,20,0,20])
    ax = fig.add_subplot(1,1,1)
    circle = plt.Circle((10,10), radius=radio, color='#0174DF', fill=False)
    ax.add_patch(circle)
    plt.scatter(x_points,y_points, c=c, marker=m)

def plot_normal(x,y,xlabel,ylabel,title, color):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x,y,c=color)

"""
Description: gets information about dataset.

Parameters:
    @param filename: name of de dataset file.
"""
def read_dataset(filename):
    dataset = open(filename, "r")
    t = []
    goal=[]

    lines=dataset.readlines()

    for line in lines:
        t.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
        goal.append([float((line.split(' ')[2]).rstrip())]) # removing /n
    dataset.close()

    return t,goal

"""
Description: average of elements in array

Parameters:
    @param x: array
"""
def avg(x):
    aux=0.0
    for i in range(len(x)):
        aux=aux+x[i]
    return (aux/len(x))

def corrida(datos, prueba, n, tasa, neuronas_intermedia):
    print("Nombre del archivo de datos: "+datos)
    x,y = read_dataset(datos)
    datos_prueba_x, datos_prueba_y = read_dataset(prueba)
    dentro=[] #dentro[0]
    fuera=[]
    aciertos=[]
    desaciertos=[]
    falso_positivo=[]
    falso_negativo=[]
    err_acum=[]
    err_entrenamiento=[]
    for i in range(n):
        net = Network([2,neuronas_intermedia,1])
        net.training(tasa, x, y)
        print("Estadistica de datos de prueba: ")
        a,b,c,d,e,f,g = net.eval_area(datos_prueba_x, datos_prueba_y)
        dentro.append(a)
        fuera.append(b)
        aciertos.append(c)
        desaciertos.append(d)
        falso_positivo.append(e)
        falso_negativo.append(f)
        err_acum.append(g)
        err_entrenamiento.append(net.get_err()[len(net.get_err())-1])
    print("\n\nPromedio de resultados para "+str(n)+" corridas:")
    print("Casos acertados: "+str(avg(aciertos))+" ,Casos no acertados: "+str(avg(desaciertos))+" ,Efectividad: "+str(avg(aciertos)*100/(avg(aciertos)+avg(desaciertos)))+"%")
    print("Error de entrenamiento: "+str(avg(err_entrenamiento))+" ,Error de prueba: "+str(avg(err_acum))+ " ,Falso Positivo: "+str(avg(falso_positivo))+" ,Falso negativo: "+str(avg(falso_negativo))+"\n")
    return avg(err_entrenamiento), avg(err_acum), avg(falso_positivo), avg(falso_negativo), avg(aciertos), avg(desaciertos), avg(aciertos)*100/(avg(aciertos)+avg(desaciertos)), dentro, fuera
#h.write("Archivo_de_datos, Archivo_de_prueba, tasa_aprendizaje, num_neurona, catidad_corridas, error_entrenamiento, error_prueba, falso_positivo, falso_negativo, casos_acertados, casos_no_acertados, efectividad\n")

# BEST LEARNING RATE -----------------------------------------------------------

alpha = [0.001,0.01,0.05,0.1,0.2] 
datos = "datosP2_AJ2018_B2_N2000.txt"
prueba = "prueba_B2_barrido_100_por_100.txt"
colores = ["#A4243B","#0174DF","#6A0888","#74DF00","#FF8000"]
c = 0


x_alpha, y_alpha = read_dataset("datosP2_AJ2018_B2_N2000.txt")
x_alpha_prueba, y_alpha_prueba = read_dataset("prueba_B2_barrido_100_por_100.txt")

h=open("result_alpha.csv", "w")
h.write("Archivo_de_datos, Archivo_de_prueba, tasa_aprendizaje, num_neurona, error_entrenamiento, error_prueba, falso_positivo, falso_negativo, casos_acertados, casos_no_acertados, efectividad\n")

for i in range(len(alpha)):
    n_alpha = Network([2,6,1])
    n_alpha.training(alpha[i],x_alpha,y_alpha)

    dentro_alpha, fuera_alpha, aciertos_alpha, desaciertos_alpha, falso_positivo_alpha, falso_negativo_alpha, err_acum_alpha = n_alpha.eval_area(x_alpha_prueba,y_alpha_prueba)
    jota_alpha = n_alpha.get_err()
    jota = []
    for i in range(len(jota_alpha)):
        jota.append(jota_alpha[i]/2000)

    print(jota)
    a,b,c,d,e,f,g,x,y = corrida(datos,prueba,1,alpha[i],6)
    iterationes = np.arange(len(jota))
    plot_normal(iterationes, jota,"Iteraciones", "J()", "Curva de Convergencia para distintos alpha",colores[c])
    h.write(datos+", "+prueba+" ,"+str(alpha[i])+" ,"+str(6)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")
    c += 1

legend1 = mpatches.Patch(color=colores[0],label="alpha = 0.001")
legend2 = mpatches.Patch(color=colores[1],label="alpha = 0.01")
legend3 = mpatches.Patch(color=colores[2],label="alpha = 0.05")
legend4 = mpatches.Patch(color=colores[3],label="alpha = 0.1")
legend5 = mpatches.Patch(color=colores[4],label="alpha = ")
plt.legend(handles=[legend1, legend2,legend3,legend4,legend5])
plt.show()
h.close()

# BEST TRAINING SET  -----------------------------------------------------------

def pruebas():
    neurons = [2,3,4,5,6,7,8,9,10]
    alpha = 0.01
    n = 10

    b1_archivos = ["datosP2_AJ2018_B1_N500.txt","datosP2_AJ2018_B1_N1000.txt","datosP2_AJ2018_B1_N2000.txt"]
    b2_archivos = ["datosP2_AJ2018_B2_N500.txt","datosP2_AJ2018_B2_N1000.txt","datosP2_AJ2018_B2_N2000.txt"]
    b1_generados_archivos = ["datos_entrenamiento_N500_B1.txt", "datos_entrenamiento_N1000_B1.txt", "datos_entrenamiento_N2000_B1.txt"]
    b2_generados_archivos = ["datos_entrenamiento_N500_B2.txt", "datos_entrenamiento_N1000_B2.txt", "datos_entrenamiento_N2000_B2.txt"]
    prueba_archivos = ["prueba_B1_barrido_100_por_100.txt","prueba_B2_barrido_100_por_100.txt"]

    h=open("result_training.csv", "w")
    h.write("Archivo_de_datos, Archivo_de_prueba, tasa_aprendizaje, num_neurona, catidad_corridas, error_entrenamiento, error_prueba, falso_positivo, falso_negativo, casos_acertados, casos_no_acertados, efectividad\n")


    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b1_generados_archivos[0],prueba_archivos[0],n,alpha,neurons[i])
        h.write(b1_generados_archivos[0]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b1_generados_archivos[1],prueba_archivos[0],n,alpha,neurons[i])
        h.write(b1_generados_archivos[1]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b1_generados_archivos[2],prueba_archivos[0],n,alpha,neurons[i])
        h.write(b1_generados_archivos[2]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b2_generados_archivos[0],prueba_archivos[1],n,alpha,neurons[i])
        h.write(b2_generados_archivos[0]+", "+prueba_archivos[1]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b2_generados_archivos[1],prueba_archivos[1],n,alpha,neurons[i])
        h.write(b2_generados_archivos[1]+", "+prueba_archivos[1]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    for i in range(len(neurons)):
        a,b,c,d,e,f,g = corrida(b2_generados_archivos[2],prueba_archivos[1],n,alpha,neurons[i])
        h.write(b2_generados_archivos[2]+", "+prueba_archivos[1]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

    h.close()

def prueba_B1():
    x_1, y_1 = read_dataset("datos_entrenamiento_N1000_B1.txt")
    n_1 = Network([2,6,1])
    n_1.training(0.01,x_1,y_1)
    x_1_prueba, y_1_prueba = read_dataset("prueba_B2_barrido_100_por_100.txt")

    dentro_1, fuera_1, aciertos_1, desaciertos_1, falso_positivo_1, falso_negativo_1, err_acum_1 = n_1.eval_area(x_1_prueba,y_1_prueba)

    jota_1 = n_1.get_err()
    iteraciones = np.arange(len(jota_1))
    plot_normal(iteraciones,jota_1,"Iteraciones", "J()", "Curva de Convergencia B1-1000, n=6","#0174DF")

    x_dentro1, y_dentro1 = get_point(dentro_1)
    x_fuera1, y_fuera1 = get_point(fuera_1)
    plot_circle_points(x_dentro1,y_dentro1,6,True)
    plot_circle_points(x_fuera1,y_fuera1,6,False)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Area B1-1000, n=6")
    plt.show()

def prueba_b2():
    x_2, y_2 = read_dataset("datos_entrenamiento_N2000_B2.txt")
    n_2 = Network([2,10,1])
    n_2.training(0.01,x_2,y_2)
    x_2_prueba, y_2_prueba = read_dataset("prueba_B2_barrido_100_por_100.txt")

    dentro_2, fuera_2, aciertos_2, desaciertos_2, falso_positivo_2, falso_negativo_2, err_acum_2 = n_2.eval_area(x_2_prueba,y_2_prueba)

    jota_2 = n_2.get_err()
    iteraciones_2 = np.arange(len(jota_2))
    plot_normal(iteraciones_2,jota_2,"Iteraciones", "J()", "Curva de Convergencia B2-2000, n=10","#0174DF")

    x_dentro2, y_dentro2 = get_point(dentro_2)
    x_fuera2, y_fuera2 = get_point(fuera_2)
    plot_circle_points(x_dentro2,y_dentro2,8,True)
    plot_circle_points(x_fuera2,y_fuera2,8,False)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Area B2-2000, n=10")
    plt.show()

