import flopy
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import uuid
import platform

if platform.system() == 'Windows':
    exe_name = './bin/mf6.exe'
else:  # Linux, macOS, etc.
    exe_name = './bin/mf6'

def modflow_model(k):
    sim_ws = os.path.join('simulation_folder', str(uuid.uuid4()))
    sim_name = 'simulation'
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name=exe_name, verbosity_level=0)

    nper = 1
    perlen  = [50.0,]
    nstp = [10,]
    tsmult = [1.2,]
    tdis_ds = list(zip(perlen, nstp, tsmult))
    time_units = "days"
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

    gwfname = 'gwf'
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True, model_nam_file="{}.nam".format(gwfname))

    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0
    imsgwf = flopy.mf6.ModflowIms(sim, 
        print_option="SUMMARY", outer_dvclose=hclose, outer_maximum=nouter,
        under_relaxation="NONE", inner_maximum=ninner, inner_dvclose=hclose,
        rcloserecord=rclose, linear_acceleration="CG", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=relax, filename="{}.ims".format(gwfname))
    sim.register_ims_package(imsgwf, [gwf.name])

    nlay = 1
    nrow, ncol = 100, 100
    delr, delc = 1.0, 1.0
    top  = 10.0
    botm = 0.0
    idomain = 1
    length_units = "meters"
    flopy.mf6.ModflowGwfdis(gwf,
        length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
        top=top, botm=botm, idomain=idomain, filename="{}.dis".format(gwfname))
    
    icelltype = 0 # 0 承压水, 1 潜水
    k11 = k
    k33 = k11 * 0.6
    flopy.mf6.ModflowGwfnpf(gwf, save_flows=False, icelltype=icelltype,
        k=k11, k33=k33,
        save_specific_discharge=True,
        filename="{}.npf".format(gwfname),
        )
    
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=0.0, sy=0.0)

    flopy.mf6.ModflowGwfic(gwf, strt=10.0, filename="{}.ic".format(gwfname))

    chd_spd = []
    for i in range(nrow):
        chd_spd.append([(0, i, 0), 10.0, 0.3])
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname='chd',
        save_flows=False,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd},
        auxiliary=["c",],
        filename="{}.chd".format(gwfname),
    )

    wel_spd = []
    for i in range(0, nrow, 2):
        wel_spd.append([(0, i, ncol-1), -5.0])
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname='wel',
        save_flows=False,
        maxbound=len(wel_spd),
        stress_period_data={0: wel_spd},
        filename="{}.wel".format(gwfname),
    )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="{}.hds".format(gwfname),
        budget_filerecord="{}.cbc".format(gwfname),
        budgetcsv_filerecord="{}.oc.csv".format(gwfname),
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # ! GWT

    gwtname = 'gwt'
    gwt = flopy.mf6.ModflowGwt(sim,
               modelname=gwtname,
               save_flows=True,
               model_nam_file="{}.nam".format(gwtname),
               )
    
    imsgwt = flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", outer_dvclose=hclose, outer_maximum=nouter,
        under_relaxation="NONE", inner_maximum=ninner, inner_dvclose=hclose,
        rcloserecord=rclose, linear_acceleration="BICGSTAB", scaling_method="NONE",
        reordering_method="NONE", relaxation_factor=relax, filename="{}.ims".format(gwtname))
    sim.register_ims_package(imsgwt, [gwt.name])

    flopy.mf6.ModflowGwtdis(gwt, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
        top=top, botm=botm, idomain=idomain, filename="{}.dis".format(gwtname))
    
    flopy.mf6.ModflowGwtic(gwt, strt=0.0, filename="{}.ic".format(gwtname))

    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM", filename="{}.adv".format(gwtname))

    al = 1.0 # Longitudinal dispersivity
    trpt = 0.1; ath1 = al * trpt # Ratio of horizontal transverse dispersivity to longitudinal dispersivity
    trpv = 0.1; atv  = al * trpv # Ratio of vertical transverse dispersivity to longitudinal dispersivity
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=al, ath1=ath1, filename="{}.dsp".format(gwtname))

    flopy.mf6.ModflowGwtmst(gwt,
        porosity=0.3,
        filename="{}.mst".format(gwtname),
    )

    sourcerecarray = [("chd", "AUX", "c")]
    flopy.mf6.ModflowGwtssm(
        gwt, 
        pname=f'ssm',
        sources=sourcerecarray, 
        filename="{}.ssm".format(gwtname),
    )

    flopy.mf6.ModflowGwtoc(gwt,
        budget_filerecord="{}.cbc".format(gwtname),
        concentration_filerecord="{}.ucn".format(gwtname),
        budgetcsv_filerecord="{}.oc.csv".format(gwtname),
        saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
        )
    
    flopy.mf6.ModflowGwfgwt(sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwtname,
        filename="{}.gwfgwt".format(sim_name)
        )
    
    sim.write_simulation(silent=True)
    sim.run_simulation(silent=True, report=False)

    head = gwf.oc.output.head().get_alldata()

    concentration = gwt.oc.output.concentration().get_alldata()

    shutil.rmtree(sim_ws)

    return head.reshape(nstp[0], 100, 100), concentration.reshape(nstp[0], 100, 100)


# ! ################################################################ ! #

# # 计算logk场的函数
# def calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, kesi):
#     kesi = kesi.reshape(1, -1)
#     logk = np.zeros((nx, ny))
#     for i_x in range(nx):
#         for i_y in range(ny):
#             logk[i_y, i_x] = mean_logk + np.sum(np.sqrt(lamda_xy) * fn_x[i_x][0] * fn_y[i_y][0] * kesi.transpose())
#     return logk

# # 观测井坐标
# wells = [
#     (5, 5), (10, 20), (15, 40), (20, 60), (25, 80),
#     (30, 10), (35, 30), (40, 50), (45, 70), (50, 90),
#     (55, 15), (60, 35), (65, 55), (70, 75), (75, 95),
#     (80, 20), (85, 40), (90, 60), (95, 80), (99, 99)
# ]

# # 模型评估函数 f(kesi) 返回观测值数组
# def f(kesi):
#     data = np.load('./data/KLE.npy', allow_pickle=True)
#     nx, ny, mean_logk, lamda_xy, fn_x, fn_y, _ = data.tolist()
#     logk = calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, kesi)
#     k = np.exp(logk).reshape(1, nx, ny)
#     head, concentration = modflow_model(k=k)
#     concentration = concentration.reshape(10, nx, ny)
#     # coords = np.array(wells)
#     # i_idx, j_idx = coords[:,0], coords[:,1]
#     return concentration#[0, i_idx, j_idx]

# # # 真值参数及对应观测
# true_kesi = np.array([
#     0.88057423, -0.19495903,  0.18768442, -0.49742327,  1.15384367,
#    -1.03706619, -0.43852673,  1.44600226, -0.15753793, -0.24283745,
#    -0.18996387,  1.59969523,  0.64235502, -0.97305024, -0.47505204,
#     1.31421505,  0.27342882,  1.29805817,  1.60486167, -0.81063640,
#     1.24518366,  2.22225735, -1.80847110,  0.48607741, -0.39719074
# ])

# test_kesi = np.array([2.05370865, -0.3014752, -0.62909858, 2.55995319, 1.36363198, -1.04075539, 
# 0.42266385, 0.12500556, 2.76703215, 2.06720309, 1.48392066, 0.23815279, 
# 0.52050699, 2.79153184, 0.64220549, -1.34400491, -1.22235897, -2.00839837, 
# -2.90618156, -0.45959112, -0.63071089, -1.23907095, -2.91552106, -1.80694558, 1.26805172])

# # data = np.load('./data/KLE.npy', allow_pickle=True)
# # nx, ny, mean_logk, lamda_xy, fn_x, fn_y, _ = data.tolist()
# # logk = calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, true_kesi)
# # plt.imshow(logk, cmap="jet")
# # plt.colorbar()
# # plt.show()

# concentration = f(true_kesi)

# # 创建 1 行 5 列的子图
# fig, axes = plt.subplots(1, 10, figsize=(15, 3), constrained_layout=True)
# im = None
# for idx, ax in enumerate(axes):
#     im = ax.imshow(concentration[idx], cmap='jet', origin='lower')
#     ax.set_title(f'time {idx+1}')  # 或者显示实际时间：f'Time = {times[idx]:.1f} d'
#     ax.axis('off')
# # 在最右边的子图下方添加一个共享的 colorbar
# cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
# cbar.set_label('c')

# plt.show()