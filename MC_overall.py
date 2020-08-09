import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import webcolors
import matplotlib.pyplot as plt


def distance(cur_atom_pos, ii, jj):
    # calculate the distance between 2 atoms: Eq. (3)
    dis = np.sqrt((cur_atom_pos[ii, 0] - cur_atom_pos[jj, 0]) ** 2 +
                  (cur_atom_pos[ii, 1] - cur_atom_pos[jj, 1]) ** 2 +
                  (cur_atom_pos[ii, 2] - cur_atom_pos[jj, 2]) ** 2)
    return dis if dis > 0 else 1e-10


def U(cur_atom_pos):
    # calculate the molecular potential energy: Eq. (2)
    U_val = 0
    for ii in range(0, atom_num - 1):
        for jj in range(ii + 1, atom_num):
            r = distance(cur_atom_pos, ii, jj)
            U_val += 4 * (1 / r ** 12 - 1 / r ** 6)  # Eq. (1)
    return U_val


def plot_atoms(atom_pos_collection, dis_str):
    # 3D representation (animation)
    figs = []
    for cur_atom_pos in atom_pos_collection:
        each_tree_scatter = []
        for ii, pos_ in enumerate(cur_atom_pos[0]):
            each_tree_scatter.append(go.Scatter3d(x=[pos_[0]], y=[pos_[1]], z=[pos_[2]],
                                                  mode='markers',
                                                  surfacecolor=colors[ii],
                                                  marker=dict(
                                                      color=colors[ii],
                                                      size=10,  # set color to an array/list of desired values
                                                      opacity=0.8,
                                                  )))
        figs.append(each_tree_scatter)
    layout = go.Layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-pos_max, pos_max], ),
            yaxis=dict(nticks=10, range=[-pos_max, pos_max], ),
            zaxis=dict(nticks=10, range=[-pos_max, pos_max], ),
            aspectmode='cube'
        ),
        title='Start Title')
    figure = {'data': figs[0],
              'layout': layout,
              'frames': []
              }
    for ii in range(len(atom_pos_collection)):
        if ii < len(atom_pos_collection) - 1:
            figure['frames'].append({'data': figs[ii],
                                     'layout': {'title': f'Epoch: {atom_pos_collection[ii][1]}, '
                                                         f'Potential: {atom_pos_collection[ii][2]}'}})
        else:
            figure['frames'].append({'data': figs[ii], 'layout': {'title': dis_str}})
    plot(figure, animation_opts={'frame': {'duration': 1000}}, filename="result(%d atoms).html" % atom_num)


# read the input data
atom_pos = []
with open('input.txt', 'r') as f:
    atom_num = int(f.readline())  # number of atoms in the molecule
    max_loop = int(f.readline())  # maximum loops
    for line in f:
        atom_pos.append(eval(line))  # initial positions of atoms

pos_max = 5  # set the region (-5, 5)
if len(atom_pos) < atom_num:
    # generate the initial positions of atoms if omitted
    for _ in range(len(atom_pos), atom_num):
        atom_pos.append((np.random.uniform(-pos_max, pos_max),  # generate the x-coordinate randomly
                         np.random.uniform(-pos_max, pos_max),  # generate the y-coordinate randomly
                         np.random.uniform(-pos_max, pos_max)))  # generate the z-coordinate randomly

atom_pos = np.array(atom_pos, dtype=np.float)
updated_epoch = 0
updated_U = np.round(U(atom_pos), 5)
total_atom_pos = [(atom_pos, updated_epoch, updated_U)]
updated = False

# choose the colors for representation of atoms in animation
if atom_num < 17:
    color_dict = webcolors.CSS2_HEX_TO_NAMES.items()
elif atom_num < 18:
    color_dict = webcolors.CSS21_HEX_TO_NAMES.items()
else:
    color_dict = webcolors.CSS3_HEX_TO_NAMES.items()
colors = []
for key, name in color_dict:
    r, g, b = webcolors.hex_to_rgb(key)
    colors.append('rgb({},{},{})'.format(r, g, b))

# set maximum loops if omitted
loops = 1000 ** (2 * atom_num) if max_loop < 0 else max_loop
print(f'Total Epoch is {loops}.\n')
record_loop = 1000  # save the molecular potential to be showed in animation per 1000 iterations
print_loop = 10000  # output the middle calculation result per 10000 iterations
loops += 1
no_change_num = 0  # denotes how much iterations the molecular potential does not improve
potentials = [U(atom_pos)]

# Monte Carlo Simulation
for i in range(1, loops):
    if i % record_loop // 10 == 0:
        potentials.append(U(atom_pos))  # to show Figure 7
    if i % record_loop == 0:
        if updated:
            total_atom_pos.append((atom_pos, updated_epoch, updated_U))
            updated = False
        if i % print_loop == 0:
            print(i, U(atom_pos))

    # original value of molecular potential
    U_old = U(atom_pos)

    # update the position of atom by Monte Carlo method
    atom_pos_new = atom_pos.copy()
    atom_ind = np.random.randint(0, atom_num)  # choose an atom randomly
    if atom_ind == atom_num:
        atom_ind -= 1
    atom_pos_new[atom_ind, 0] = np.random.uniform(-pos_max, pos_max)  # change the x-coordinate of the chosen atom randomly in the region (-5, 5)
    atom_pos_new[atom_ind, 1] = np.random.uniform(-pos_max, pos_max)  # change the y-coordinate of the chosen atom randomly in the region (-5, 5)
    atom_pos_new[atom_ind, 2] = np.random.uniform(-pos_max, pos_max)  # change the z-coordinate of the chosen atom randomly in the region (-5, 5)

    # new value of molecular potential
    U_new = U(atom_pos_new)
    delta_U = U_new - U_old  # difference in molecular potential
    if delta_U < 0:
        atom_pos = atom_pos_new.copy()  # accept the new position
        updated = True
        updated_epoch = i + 1
        updated_U = np.round(U_new, 5)
        no_change_num = 0
    else:  # reject the new position
        no_change_num += 1
        if no_change_num > print_loop * 10:
            break  # if the molecular potential is not improved for 100000 iterations, the calculation is terminated.
print('finished calculation!')

# output file
file = open('output.txt', 'w')
file.write(f'Number of atoms: {atom_num}\n')
file.write(f'Total Potential: {np.round(U(atom_pos), 5)} \t(unit: 1.67e-21 J)')
for i, pos in enumerate(atom_pos):
    file.write("{}-th atom's position is: ({}, {}, {})\n".format(i + 1, pos[0], pos[1], pos[2]))

distance_str = f'Number of atoms: {atom_num}<br>'
distance_str += f'Total Potential: {np.round(U(atom_pos), 5)} \t(unit: 1.67e-21 J)'
for i in range(0, atom_num - 1):
    for j in range(i + 1, atom_num):
        r = distance(atom_pos, i, j)
        print_str = f'{i + 1} - {j + 1} distance: {np.round(r, 5)}'
        file.write('\n' + print_str)
        distance_str += '<br>' + print_str
distance_str += ' \t(unit: 3.4e-10 m)'
file.write(' \t(unit: 3.4e-10 m)\n')
file.close()

# figure 7
plt.plot(potentials)
plt.xlabel('epoch / %d' % record_loop)
plt.ylabel('Molecular Potential Energy (unit: 1.67e-21 J)')
plt.title('Change in Molecular potential energy with epoch')
plt.savefig('Molecular Potential.png')
plt.show()

# 3D representation (animation)
plot_atoms(total_atom_pos, distance_str)
