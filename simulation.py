from netpyne import specs, sim
import matplotlib.pyplot as plt
import numpy as np
import neuron

# Network parameters
netParams = specs.NetParams()  # object of class NetParams to store the network parameters

## Cell parameters
netParams.cellParams['pyr'] = {
    'secs': {
        'soma': {
            'geom': {
                'diam': 18.8,   
                'L': 18.8, 
                'Ra': 123.0},
            'mechs': {
                'hh': {
                    'gnabar': 0.12, 
                    'gkbar': 0.036, 
                    'gl': 0.0003, 
                    'el': -70}
            }
        },
        'dend1': {
            'geom': {
                'diam': 5.0, 
                'L': 75.0, 
                'Ra': 150.0, 
                'cm': 1
            }, 
            'mechs' :{
                'pas':{
                    'g': 0.0000357, 
                    'e': -70}
            },
            'topol' : {
                'parentSec': 'soma', 
                'parentX': 1.0, 
                'childX': 0
                }
        },
        'dend2': {
            'geom': {
                'diam': 5.0, 
                'L': 75.0, 
                'Ra': 150.0, 
                'cm': 1
            }, 
            'mechs' :{
                'pas':{
                    'g': 0.0000357, 
                    'e': -70}
            },
            'topol' : {
                'parentSec': 'dend1', 
                'parentX': 1.0, 
                'childX': 0
                }
            
        }
    },
    "pt3d": True
}

## Population parameters
netParams.popParams['E'] = { #xyzRange = [0, 100]
    'cellType': 'pyr', 
    'numCells': 80}

netParams.popParams['I'] = {
    'cellType': 'pyr', 
    'numCells': 20}

## Synaptic mechanism parameters
netParams.synMechParams['exc'] = {
    'mod': 'Exp2Syn', 
    'tau1': 0.1, 
    'tau2': 5.0, 
    'e': 0}  # excitatory synaptic mechanism

netParams.synMechParams['inh'] = {
    'mod': 'Exp2Syn', 
    'tau1': 0.1, 
    'tau2': 5.0, 
    'e': -80}  # excitatory synaptic mechanism

# Stimulation parameters
netParams.stimSourceParams['bkg'] = {
    'type': 'NetStim', 
    'rate': 1, #hz 
    'noise': 0.5}

netParams.stimTargetParams['bkg->E'] = {
    'source': 'bkg', 
    'conds': {'pop': 'E'}, 
    'weight': 0.01, 
    'delay': 1,
    'synMech': 'exc'}

## Connectivity rules
netParams.connParams['E->E'] = {    #  label
    'preConds': {'pop': 'E'},       # conditions of presyn cells
    'postConds': {'pop': 'E'},      # conditions of postsyn cells
    'divergence': 5,                # probability of connection
    'weight': 0.01,                 # synaptic weight
    'delay': 5,                     # transmission delay (ms)
    'synMech': 'exc'}               # synaptic mechanism

netParams.connParams['E->I'] = {    #  label
    'preConds': {'pop': 'E'},       # conditions of presyn cells
    'postConds': {'pop': 'I'},      # conditions of postsyn cells
    'divergence': 5,                # probability of connection
    'weight': 0.01,                 # synaptic weight
    'delay': 5,                     # transmission delay (ms)
    'synMech': 'exc'}               # synaptic mechanism

netParams.connParams['I->E'] = {    #  label
    'preConds': {'pop': 'I'},       # conditions of presyn cells
    'postConds': {'pop': 'E'},      # conditions of postsyn cells
    'divergence': 10,               # probability of connection
    'weight': 0.01,                 # synaptic weight
    'delay': 5,                     # transmission delay (ms)
    'synMech': 'inh'}               # synaptic mechanism

simConfig = specs.SimConfig()       # object of class SimConfig to store simulation configuration

# #------------------------------------------------------------------------------
# #  extracellular mechs
# #------------------------------------------------------------------------------
for celltyp in netParams.cellParams.keys():
    label = []
    for secname in netParams.cellParams[celltyp]['secs'].keys():
        netParams.cellParams[celltyp]['secs'][secname]['mechs']['extracellular'] = {}

# Simulation options

simConfig.duration = 3000          # Duration of the simulation, in ms
simConfig.dt = 0.01                # Internal integration timestep to use
simConfig.verbose = False           # Show detailed messages
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordCells = ['E', 'I']
simConfig.recordStep = 0.1          # Step size in ms to save data (eg. V traces, LFP, etc)
simConfig.filename = 'raster'  # Set file output name
simConfig.saveJson = False
simConfig.recordLFP = [[50,50,50]]

#simConfig.analysis['plotTraces'] = {'include': ['E', 'I'], 'saveFig': True}  # Plot recorded traces for this list of cells
# simConfig.analysis['plotRaster'] = {'showFig': True}                  # Plot a raster
# simConfig.analysis['plotSpikeHist'] = {'include': ['E', 'I'], 'showFig': True}
# simConfig.analysis['plot'] = {'saveFig': False}                   # plot 2D cell positions and connections
# simConfig.analysis['plotRateSpectrogram'] = {'include': ['all'], 'saveFig': True}
#simConfig.analysis['recordLFP'] = [[50,50,50]]

sim.initialize(
    simConfig = simConfig, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.net.defineCellShapes()

### ---------------------------------------------------- ###
### setting the extracelular stimulation at every cell   ###
### ---------------------------------------------------- ###

# The parameters of the extracellular point current source
acs_params = {'position': [20.0, 20.0, 20.0],  # um
              'amp': 50.,  # uA,
              'stimstart': 1200,  # ms
              'stimend': 2200,  # ms
              'frequency': 50,  # Hz
              'sigma': 0.57  # decay constant S/m
              }

skull_attenuation = 0.01*710 #conductivity of bone(S/m) * thickness of rat skull um

def getDend1Pos(cell):
        """
        Get soma position;
        Used to calculate seg coords for LFP calc (one per population cell; assumes same morphology)
        """

        n3dsoma = 0
        r3dsoma = np.zeros(3)
        for sec in [sec for secName, sec in cell.secs.items() if 'dend1' in secName]:
            sec['hObj'].push()
            n3d = int(neuron.h.n3d())  # get number of n3d points in each section
            r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsoma += n3d

            for i in range(n3d):
                r3dsoma[0] += neuron.h.x3d(i)
                r3dsoma[1] += neuron.h.y3d(i)
                r3dsoma[2] += neuron.h.z3d(i)

            neuron.h.pop_section()

        r3dsoma /= n3dsoma

        return r3dsoma

def getDend2Pos(cell):
        """
        Get soma position;
        Used to calculate seg coords for LFP calc (one per population cell; assumes same morphology)
        """

        n3dsoma = 0
        r3dsoma = np.zeros(3)
        for sec in [sec for secName, sec in cell.secs.items() if 'dend2' in secName]:
            sec['hObj'].push()
            n3d = int(neuron.h.n3d())  # get number of n3d points in each section
            r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsoma += n3d

            for i in range(n3d):
                r3dsoma[0] += neuron.h.x3d(i)
                r3dsoma[1] += neuron.h.y3d(i)
                r3dsoma[2] += neuron.h.z3d(i)

            neuron.h.pop_section()

        r3dsoma /= n3dsoma

        return r3dsoma

def insert_v_ext(cell, v_ext, t_ext):

    cell.t_ext = neuron.h.Vector(t_ext)
    cell.v_ext = []

    for s in v_ext:
        cell.v_ext.append(neuron.h.Vector(s))
        
    # play v_ext into e_extracellular reference
    i = 0
    cell.v_ext[i].play(cell.secs['soma']['hObj'](0.5)._ref_e_extracellular, cell.t_ext)

def make_extracellular_stimuli(acs_params, cell, numsetions=3):
    """ Function to calculate and apply external potential """
    x0, y0, z0 = acs_params['position']
    ext_field = np.vectorize(lambda x, y, z: 1 / (4 * np.pi *
                                                  (acs_params['sigma'] * 
                                                   np.sqrt((x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2) + skull_attenuation)))

    stimstart = acs_params['stimstart']
    stimend = acs_params['stimend']
    stimdif = stimend-stimstart

    # MAKING THE EXTERNAL FIELD
    n_tsteps = int(stimdif / simConfig.dt + 1)
    n_start = int(stimstart/simConfig.dt)
    n_end = int(stimend/simConfig.dt + 1)
    t = np.arange(start=n_start, stop=n_end) * simConfig.dt
    pulse = acs_params['amp'] * 1000. * \
          np.sin(2 * np.pi * acs_params['frequency'] * t / 1000)
        
        
    v_cell_ext = np.zeros((numsetions, n_tsteps))
    
    v_cell_ext[:, :] = (ext_field(np.array([cell.getSomaPos()[0], abs(cell.getSomaPos()[1]), cell.getSomaPos()[2]]), 
                                 np.array([getDend1Pos(cell)[0], getDend1Pos(cell)[1], getDend1Pos(cell)[2]]), 
                                 np.array([getDend2Pos(cell)[0], getDend2Pos(cell)[1], getDend2Pos(cell)[2]])).reshape(numsetions, 1)
                                 * pulse.reshape(1, n_tsteps))
    
    insert_v_ext(cell, v_cell_ext, t)

    return ext_field, pulse

#Add extracellular stim
for c,metype in enumerate(sim.net.cells):
    if 'presyn' not in metype.tags['pop']:
        ext_field, pulse = make_extracellular_stimuli(acs_params, sim.net.cells[c])

sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#

### ------------------------
### saving the data 
### ------------------------

import os
os.chdir(path='.\data')

spkid = np.array(sim.simData['spkid'].to_python())
spktime = np.array(sim.simData['spkt'].to_python())
lfpdata = sim.simData['LFP']

np.savetxt(f"spkid_{acs_params['frequency']}Hz.csv", spkid, delimiter=",")
np.savetxt(f"spktime_{acs_params['frequency']}Hz.csv", spktime, delimiter=",")
np.savetxt(f"lfpdata_{acs_params['frequency']}Hz.csv", lfpdata, delimiter=",")

## LFP spectrogram
sim.analysis.plotLFP(electrodes=[0], saveFig= f".\spectrogram_{acs_params['frequency']}Hz.png", plots='spectrogram')

## plotting the raster, LFP and stimulus
fig = plt.figure()
grid = plt.GridSpec(3, 1,height_ratios=[2,1.5,1], hspace=0.03)

ax_0 = fig.add_subplot(grid[0])
ax_0.plot(spktime[spkid<80], spkid[spkid<80],'|', markersize=0.8)
ax_0.plot(spktime[spkid>=80], spkid[spkid>=80],'|' , markersize=0.8)
ax_0.set_xticklabels([])
ax_0.set_xticks([])
ax_0.spines['bottom'].set_visible(False)
ax_0.set_ylabel('Neuron ID')

ax_1 = fig.add_subplot(grid[1])
ax_1.plot(np.arange(0,3000, 0.1),lfpdata)
ax_1.set_xticklabels([])
ax_1.set_xticks([])
ax_1.set_ylim([-0.29,0.17])
ax_1.spines[['bottom','top']].set_visible(False)
ax_1.set_ylabel('LFP (mV)')

ax_2 = fig.add_subplot(grid[2])
pulse = acs_params['amp'] * 1000. * \
          np.sin(2 * np.pi * acs_params['frequency'] * np.arange(0,3000) / 1000)
pulse[:acs_params['stimstart']] = 0
pulse[acs_params['stimend']:] = 0
ax_2.plot(pulse*0.001)
ax_2.spines['top'].set_visible(False)
ax_2.set_ylabel('Stimulus ($\mu$A)')
ax_2.set_xlabel('Time (ms)')

fig.savefig(fname=f"raster_LFP_Stim_{acs_params['frequency']}Hz")