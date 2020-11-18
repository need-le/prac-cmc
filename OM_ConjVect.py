import numpy as np
from math import sqrt
from tkinter import *
from tkinter import scrolledtext
from tkinter.ttk import Checkbutton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
notations = True
dim = 4
eps = 0.001
u_0 = [4.0, -2.0, 4.0, 4.0]

bg_color = 'white'
inserttext_bg_color = 'white'
console_bg_color = 'azure'

func_str = "J(u) = 20(u[2] - u[0])^2 + 50(u[3] - u[1])^2 + 3u[1]^2 - 2u[0]u[1] + u[0]^2"

J_test = lambda u: (u[0] + 3)**2 + 3*(u[1] - 1)**2 + 4*(u[2] - 2)**2  - 3

J = lambda u: 20*(u[2] - u[0])**2 + 50*(u[3] - u[1])**2 + 3*u[1]**2 - 2*u[0]*u[1] + u[0]**2

def minimize (f, a = 0.0, c = 1.0,  precision = 0.0001): 
	"""
	Метод покрытий для минимизации функции одной переменной
	"""
	#eps = precision**2
	delta = (c - a)*precision
	x = a
	values = []
	while x <= c:
		values.append(f(x))
		x += delta
	return a + delta*values.index(min(values))
	
def Gradient(func, point):
	grad = []
	h = 0.0001
	for pos, num in enumerate(point):
		dim_f = np.array(point)
		dim_b = np.array(point)
		dim_f[pos] += h
		dim_b[pos] -= h
		grad.append((func(dim_f) - func(dim_b)) / (2 * h))
	return np.array(grad)

	
def conj_vect(J, eps, u_0, steps_max, txt, ax, graph):
	"""
	Метод сопряженных векторов для минимизации функции многих переменных

	u_k+1 = u_k - alpha_k*p_k
		,где: 
	u_0 - начальное приближение
	p_0 = grad(J)(u_0)
	p_k = grad(J)(u_k) - beta_k*u_(k-1) 
	
	alpha_k >= 0 ; 
	alpha_k = argmin(f_k(alpha) | alpha > 0), 
		where f_k(alpha) = J(u_k - alpha*p_k)
		
	beta_k = - |grad(J)(u_k)|**2/
				|grad(J)(u_k-1)|**2
				
	"""
	clmns = ["alpha_k"] + ["beta_k"] + [f"u_k{i}" for i in range(dim)] + [f"p_k{i}" for i in range(dim)] + ["J_k"]
	u_old = np.array(u_0)
	p = Gradient(J, u_old)
	f = lambda alph: J(u_old - alph*p)
	alpha = minimize(f, 0, 2, 0.000001)
	beta = 0.
	k = 1
	vals_arr = [J(u_old)]
	iter_data = [alpha] + [beta] + u_old.tolist() + p.tolist() + [J(u_old)]
	df = [iter_data]
	
	if (notations == True):
		txt.insert(INSERT, "Шаг 1\n")
		txt.insert(INSERT, "  Alpha | Beta  - ")
		txt.insert(INSERT, alpha)
		txt.insert(INSERT, " | ") 
		txt.insert(INSERT, beta)
		txt.insert(INSERT, "\n Значение аргумента - ")
		txt.insert(INSERT, u_old)
		txt.insert(INSERT, "\n Значение направления - ")
		txt.insert(INSERT, p)
	
	while(np.linalg.norm(Gradient(J, u_old))>= eps) and (k < steps_max):
		
		u_new = u_old - alpha*p

		if (k % 4 == 0):
			beta = 0.
		else:
			beta = - np.linalg.norm(Gradient(J, u_new))**2 / (np.linalg.norm(Gradient(J, u_old))**2)
		p = Gradient(J, u_new) - beta*p
		f = lambda alph: J(u_old - alph*p)
		alpha = minimize(f, 0, 2, 0.0001)
		u_old = u_new
		
		vals_arr.append(J(u_old))
		
		iter_data = [alpha] + [beta] + u_old.tolist() + p.tolist() + [J(u_old)]
		df.append(iter_data)
		
		k += 1
		
		ax.cla()
		ax.set_ylim([0.0 , max(vals_arr)]) 
		ax.grid()
		ax.plot(range(1, k + 1), vals_arr, color = 'red')
		graph.draw()
		
		if (notations == True):
			txt.insert(INSERT, "\nШаг ")
			txt.insert(INSERT, k)
			txt.insert(INSERT, "\n  Alpha -               ")
			txt.insert(INSERT, alpha)
			txt.insert(INSERT, "\n  Beta  -               ")
			txt.insert(INSERT, beta)
			txt.insert(INSERT, "\n Значение аргумента -   ")
			txt.insert(INSERT, u_old)
			txt.insert(INSERT, "\n Значение направления - ")
			txt.insert(INSERT, p)
			txt.insert(INSERT, "\n Минимум функции:       ")
			txt.insert(INSERT, J(u_old))

	exceldata = pd.DataFrame(df, columns = clmns)
	try:
		exceldata.to_excel("Results.xlsx")
	except:
		pass
		
	ax.cla()
	ax.set_ylim([0.0 , max(vals_arr)]) 
	ax.grid()
	ax.plot(range(1, k + 1), vals_arr, color = 'red')
	graph.draw()
	
	txt.insert(INSERT, "\n\n Результат получен через ") 
	txt.insert(INSERT, k)
	txt.insert(INSERT, "шагов")
	txt.insert(INSERT, "\n Значение аргумента:")
	txt.insert(INSERT, np.array(u_old))
	txt.insert(INSERT, "\n Минимум функции: ")
	txt.insert(INSERT, J(u_old))
	txt.update_idletasks()
		
	#return(u_new)


def clicked():
	ax.cla()
	ax.grid()
	graph.draw()
	
	if (len(txt_2.get()) != 0):
		eps = float(txt_2.get())
	else:
		eps = 0.001
	
	if (len(txt_3.get()) != 0):
		steps_max = float(txt_3.get())
	else:
		steps_max = 1000
	
	if (chk4_state.get() == 1) or (len(txt_4.get()) == 0):
		u_0 = np.array([np.random.uniform(-1, 1, dim)]).reshape(dim)
	else:
		u_0 = list(map(float, txt_4.get().split()))
	
	scroll_txt.delete(1.0, END)
	conj_vect(J, eps, u_0, steps_max, scroll_txt, ax, graph)
	
	return
	
window = Tk()
window["bg"] = bg_color
window.title ("Практикум на ЭВМ")
window.geometry('1600x600')

lbl_1 = Label(window, text = "Функционал", font = ("Cambria", 20))
lbl_1.configure(background = bg_color)
lbl_1.grid(column = 0, row = 0)

lbl_1_func = Label(window, text = func_str, font = ("Cambria", 15))
lbl_1_func.configure(background = bg_color)
lbl_1_func.grid(column = 0, row = 1)

btn = Button(window, text = "Минимизировать", bg = bg_color, command = clicked)
btn.configure(background = inserttext_bg_color)
btn.grid(column = 0, row = 2)

lbl_2 = Label(window, text = "Точность", font = ("Cambria", 14))
lbl_2.grid(column = 1, row = 0, sticky = 'W')

txt_2 = Entry(window, width = 15)
txt_2.configure(background = inserttext_bg_color)
txt_2.insert(0, '0.001')
txt_2.grid(column = 1, row = 0)

lbl_3 = Label(window, text = "Кол-во итераций", font = ("Cambria", 14))
lbl_3.grid(column = 1, row = 1, sticky = 'W')

txt_3 = Entry(window, width = 15)
txt_3.configure(background = inserttext_bg_color)
txt_3.insert(0, '228')
txt_3.grid(column = 1, row = 1)


lbl_4 = Label(window, text = "Начальная точка", font = ("Cambria", 14))
lbl_4.grid(column = 1, row = 2, sticky = 'W')

txt_4 = Entry(window, width = 15)
txt_4.configure(background = inserttext_bg_color)
txt_4.insert(0, '22 8 14 88')
txt_4.grid(column = 1, row = 2)

chk4_state = IntVar()
chk4_state.set(1)
chk4 = Checkbutton(window, text = "Автоматически", var = chk4_state)
chk4.grid(column = 1, row = 2, sticky = 'E')

scroll_txt = scrolledtext.ScrolledText(window, width = 100, height = 27, pady = 5, padx = 5)
scroll_txt.configure(background = console_bg_color)
scroll_txt.grid(column = 0, row = 3)

fig = Figure()     
ax = fig.add_subplot(111) 
ax.set_xlabel("Номер шага") 
ax.set_ylabel("Значение J(u)") 
ax.grid()

graph = FigureCanvasTkAgg(fig, master = window)
graph.get_tk_widget().grid(column = 1, row = 3, pady = 5, padx = 5)

window.mainloop()

	
	
	
	
	
	
	
	
	
	
	