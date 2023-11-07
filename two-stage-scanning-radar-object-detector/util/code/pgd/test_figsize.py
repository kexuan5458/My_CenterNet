import matplotlib.pyplot as plt

plt.scatter([0], [0])

boundary = 0.175*1142/2
plt.xlim(-boundary,boundary)
plt.ylim(-boundary,boundary)

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
						hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.gca().set_aspect('equal', adjustable='box')

fig = plt.gcf()
DPI = fig.get_dpi()
# import ipdb; ipdb.set_trace(context=7)


fig.set_size_inches(1142.0/float(DPI),1142.0/float(DPI))
plt.savefig('test.png', format = "PNG", bbox_inches = 'tight', pad_inches = 0, transparent=False, dpi=DPI)