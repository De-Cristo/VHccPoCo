from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
import vjet_weights
from vjet_weights import *

import MVA
from MVA.gnnmodels import GraphAttentionClassifier

import click

import workflow_VHqq
from workflow_VHqq import VHqqBaseProcessor

import CommonSelectors
from CommonSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_VHqq)
cloudpickle.register_pickle_by_value(CommonSelectors)
cloudpickle.register_pickle_by_value(vjet_weights)
cloudpickle.register_pickle_by_value(MVA)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/ctagging.yaml",
                                                  f"{localdir}/params/btagger.yaml",
                                                  f"{localdir}/params/trainings.yaml",
                                                  f"{localdir}/params/trainings_Hbb.yaml",
                                                  update=True)
files_2016 = [
    f"{localdir}/datasets/Run2UL2016_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2016_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2016_DATA.json",
]
files_2017 = [
    f"{localdir}/datasets/Run2UL2017_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2017_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2017_DATA.json",
]
files_2018 = [
    f"{localdir}/datasets/Run2UL2018_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2018_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2018_DATA.json",
]
files_Run3 = [
    f"{localdir}/datasets/Run3_MC_VJets.json",
    f"{localdir}/datasets/Run3_MC_OtherBkg.json",
    f"{localdir}/datasets/Run3_DATA.json",
    f"{localdir}/datasets/Run3_MC_Sig.json",
]

parameters["proc_type"] = "ZNuNu"
parameters["save_arrays"] = True
parameters["separate_models"] = False
parameters["run_bdt"] = False
parameters['run_dnn'] = False
parameters['run_gnn'] = False
ctx = click.get_current_context()
outputdir = ctx.params.get('outputdir')

outVariables = [
    "nJet", "EventNr", "ZHbb_pt_ratio", "ZHbb_deltaPhi", 
    "deltaPhi_jet1_MET", "deltaPhi_b1_MET", "deltaPhi_b2_MET",
    "dibjet_m", "dibjet_pt", "dibjet_dr", "dibjet_deltaPhi", "dibjet_deltaEta",
    "dibjet_pt_max", "dibjet_pt_min", "dibjet_mass_max", "dibjet_mass_min",
    "dibjet_BvsL_max", "dibjet_BvsL_min", "VHbb_pt_ratio", "VHbb_deltaPhi",
    "JetGood_Ht", "Z_pt", "Z_phi", "Z_eta"
]

btagger = 'PNet'

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                # "DATA_MET",
                # "WW",
                # "WZ",
                # "ZZ",
                #"QCD",
                # "ZJetsToNuNu_HT_MLM",
                # "ZJetsToNuNu_1JPT_FxFx",
                # "ZJetsToNuNu_2JPT_FxFx",
                # "ZJetsToNuNu_2JPT_FxFx_low",
                # "ZJetsToNuNu_2JPT_FxFx_high",
                # "ZJetsToNuNu_2JPT_FxFx_12",
                # "ZJetsToNuNu_2JPT_FxFx_24",
                # "ZJetsToNuNu_2JPT_FxFx_46",
                # "WJetsToLNu_MLM",
                # "WJetsToLNu_FxFx",
                # "WJetsToQQ_MLM",
                "TTTo2L2Nu",
                "TTToSemiLeptonic",
                "TTToHadrons",
                "SingleTop",
                # "WH_Hto2C_WtoLNu",
                # "ZH_Hto2C_Zto2Nu",
                # "ggZH_Hto2C_Zto2Nu",
                # "ZH_Hto2B_Zto2Nu",
                # "ggZH_Hto2B_Zto2Nu"
            ],
            
            "samples_exclude" : [],
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_preEE','2022_postEE']
            # "year": ['2022_preEE']
        },

        "subsamples": subsampleDict

    },

    workflow = VHqqBaseProcessor,
    workflow_options = {"dump_columns_as_arrays_per_chunk": 
                        "root://eosuser.cern.ch//eos/user/l/lichengz/BDT_train_VHqq_ZNuNu_TOP_1221/"} if parameters["save_arrays"] else {},
    
    # workflow_options = {"dump_columns_as_arrays_per_chunk": 
    #                     "./PourTest/"} if parameters["save_arrays"] else {},
    
    skim = [get_HLTsel(primaryDatasets=["MET"]),
            get_nObj_min(2, 32., "Jet"),
            get_JetVetoMap(),
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [ZNNHBB_2J()],
  
    categories = {
      # "baseline_ZNNHBB_2J_nn": [passthrough],
      
      "SR_MET2B_nn": [ZNuNuHBB_2J_Common_Selectors(pt_met=180, add_lep=0, pt_dibjet=150, 
                                                   b1_pt=60, b2_pt=35, VHdPhi=2.2,
                                                   HVpTRatio_min=0.5, HVpTRatio_max=2,
                                                   b1_mass_min = 5, b1_mass_max=30,
                                                   b2_mass_min = 5, b2_mass_max=30
                                                  ), 
                      nAddJetCut(1, False), dijet_mass(90, 250, False), 
                      bjet_tagger(f'{btagger}', 0, 'T', False), bjet_tagger(f'{btagger}', 1, 'M', False)],
      
      "CR_LF_nn": [ZNuNuHBB_2J_Common_Selectors(pt_met=180, add_lep=0, pt_dibjet=150, 
                                                   b1_pt=60, b2_pt=35, VHdPhi=2.2,
                                                   HVpTRatio_min=0.5, HVpTRatio_max=2,
                                                   b1_mass_min = 5, b1_mass_max=30,
                                                   b2_mass_min = 5, b2_mass_max=30
                                                  ), 
                   nAddJetCut(1, False), dijet_mass(50, 250, False), 
                   bjet_tagger(f'{btagger}', 0, 'M', True), bjet_tagger(f'{btagger}', 1, 'L', True)],
      
      "CR_B_nn": [ZNuNuHBB_2J_Common_Selectors(pt_met=180, add_lep=0, pt_dibjet=150, 
                                                   b1_pt=60, b2_pt=35, VHdPhi=2.2,
                                                   HVpTRatio_min=0.5, HVpTRatio_max=2,
                                                   b1_mass_min = 5, b1_mass_max=30,
                                                   b2_mass_min = 5, b2_mass_max=30
                                                  ), 
                  nAddJetCut(1, False), dijet_mass(90, 250, False), 
                  bjet_tagger(f'{btagger}', 0, 'T', False), bjet_tagger(f'{btagger}', 1, 'M', True)],
      
      "CR_BB_nn": [ZNuNuHBB_2J_Common_Selectors(pt_met=180, add_lep=0, pt_dibjet=150, 
                                                   b1_pt=60, b2_pt=35, VHdPhi=2.2,
                                                   HVpTRatio_min=0.5, HVpTRatio_max=2,
                                                   b1_mass_min = 5, b1_mass_max=30,
                                                   b2_mass_min = 5, b2_mass_max=30
                                                  ), 
                   nAddJetCut(1, False), dijet_mass(50, 250, False), dijet_mass(90, 150, True), 
                   bjet_tagger(f'{btagger}', 0, 'T', False), bjet_tagger(f'{btagger}', 1, 'M', False)],
      
      "CR_TT_nn": [ZNuNuHBB_2J_Common_Selectors(pt_met=180, add_lep=0, pt_dibjet=150, 
                                                   b1_pt=60, b2_pt=35, VHdPhi=2.2,
                                                   HVpTRatio_min=0.5, HVpTRatio_max=2,
                                                   b1_mass_min = 5, b1_mass_max=30,
                                                   b2_mass_min = 5, b2_mass_max=30
                                                  ), 
                   nAddJetCut(2, False), dijet_mass(90, 250, False), min_dPhi_bJ_MET_1p57, 
                   bjet_tagger(f'{btagger}', 0, 'T', False), bjet_tagger(f'{btagger}', 1, 'M', False)],
    },
  
    
    columns = {
        "common": {
            "bycategory": {
                    # "baseline_ZNNHBB_2J_nn": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_MET2B_nn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_nn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_nn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_nn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_nn": [ ColOut("events", outVariables, flatten=False), ]
                }
         },
    },

    weights = {
        "common": {
            "inclusive": ["signOf_genWeight","lumi","XS",
                          # "pileup", #Not in 2022/2023
                          # "sf_mu_id","sf_mu_iso",
                          # "sf_ele_reco","sf_ele_id",
                          #"sf_ctag", "sf_ctag_calib"
                          ],
            "bycategory" : {
                #"baseline_2L2J_ctag" : ["sf_ctag"],
                #"baseline_2L2J_ctag_calib": ["sf_ctag","sf_ctag_calib"]
            }
        },
        "bysample": {
            # "ZJetsToNuNu_NJPT_FxFx": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx_low": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx_high": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx_12": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx_24": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_2JPT_FxFx_46": {"inclusive": ["weight_vjet"] },
            # "ZJetsToNuNu_1JPT_FxFx": {"inclusive": ["weight_vjet"] },
            # "WJetsToLNu_FxFx": {"inclusive": ["weight_vjet"] },
        },
    },
    
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    # "pileup",
                    # "sf_mu_id", "sf_mu_iso",
                    # "sf_ele_reco", "sf_ele_id",
                    #"sf_ctag",
                ]
            },
            "bysample": { }
        },
        #"shape": {
        #    "common":{
        #        #"inclusive": [ "JES_Total_AK4PFchs", "JER_AK4PFchs" ] # For Run2UL
        #        "inclusive": [ "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi" ] # For Run3
        #    }
        #}
    },

    variables = {
      
        # **lepton_hists(coll="LeptonGood", pos=0),
        # **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),

        "dijet_m" : HistConf( [Axis(field="dijet_m", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_m_zoom" : HistConf( [Axis(field="dijet_m", bins=50, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),

        "dijet_pt" : HistConf( [Axis(field="dijet_pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_pt_zoom" : HistConf( [Axis(field="dijet_pt", bins=50, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),

        "dijet_dr" : HistConf( [Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_dr_zoom" : HistConf( [Axis(field="dijet_dr", bins=25, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),

        "dijet_deltaPhi": HistConf( [Axis(field="dijet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{jj}$")] ),
        "dijet_deltaPhi_zoom": HistConf( [Axis(field="dijet_deltaPhi", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{jj}$")] ),

        "dijet_deltaEta": HistConf( [Axis(field="dijet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{jj}$")] ),
        "dijet_deltaEta_zoom": HistConf( [Axis(field="dijet_deltaEta", bins=25, start=0, stop=4, label=r"$\Delta \eta_{jj}$")] ),

        "Z_pt": HistConf( [Axis(field="Z_pt", bins=100, start=0, stop=400, label=r"$p_T{Z}$ [GeV]")] ),
        "Z_pt_zoom": HistConf( [Axis(field="Z_pt", bins=25, start=0, stop=400, label=r"$p_T{Z}$ [GeV]")] ),
        "Z_phi": HistConf( [Axis(field="Z_phi",  bins=64, start=-math.pi, stop=math.pi, label=r"$phi_{Z}$")] ),
        "Z_phi_zoom": HistConf( [Axis(field="Z_phi",  bins=16, start=-math.pi, stop=math.pi, label=r"$phi_{Z}$")] ),
        
        "dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
        "dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
        
        "ZHbb_pt_ratio": HistConf( [Axis(field="ZHbb_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(H)}{p_T(Z)}$")] ),
        "ZHbb_pt_ratio_zoom": HistConf( [Axis(field="ZHbb_pt_ratio", bins=20, start=0, stop=2, label=r"$\frac{p_T(H)}{p_T(Z)}$")] ),
        "ZHbb_deltaPhi": HistConf( [Axis(field="ZHbb_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (Z, H)$")] ),
        "ZHbb_deltaPhi_zoom": HistConf( [Axis(field="ZHbb_deltaPhi", bins=20, start=0, stop=math.pi, label=r"$\Delta \phi (Z, H)$")] ),

        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=900, label=r"Jet HT [GeV]")] ),
        "HT_zoom":  HistConf( [Axis(field="JetGood_Ht", bins=25, start=0, stop=900, label=r"Jet HT [GeV]")] ),
        # "met_pt": HistConf( [Axis(coll="PuppiMET", field="pt", bins=50, start=100, stop=600, label=r"PuppiMET $p_T$ [GeV]")] ),
        # "met_phi": HistConf( [Axis(coll="PuppiMET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"PuppiMET $phi$")] ),

        "met_deltaPhi_j1": HistConf( [Axis(field="deltaPhi_jet1_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, cjet 1)")] ),
        "met_deltaPhi_j2": HistConf( [Axis(field="deltaPhi_jet2_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, cjet 2)")] ),
        
        "met_deltaPhi_b1": HistConf( [Axis(field="deltaPhi_b1_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, bjet 1)")] ),
        "met_deltaPhi_b2": HistConf( [Axis(field="deltaPhi_b2_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, bjet 2)")] ),
        
        "dibjet_m" : HistConf( [Axis(field="dibjet_m", bins=100, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
        "dibjet_m_zoom" : HistConf( [Axis(field="dibjet_m", bins=50, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),

        "dibjet_pt" : HistConf( [Axis(field="dibjet_pt", bins=100, start=0, stop=400, label=r"$p_T{bb}$ [GeV]")] ),
        "dibjet_pt_zoom" : HistConf( [Axis(field="dibjet_pt", bins=50, start=0, stop=400, label=r"$p_T{bb}$ [GeV]")] ),

        "dibjet_dr" : HistConf( [Axis(field="dibjet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),
        "dibjet_dr_zoom" : HistConf( [Axis(field="dibjet_dr", bins=25, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),

        "dibjet_deltaPhi": HistConf( [Axis(field="dibjet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{bb}$")] ),
        "dibjet_deltaPhi_zoom": HistConf( [Axis(field="dibjet_deltaPhi", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{bb}$")] ),

        "dibjet_deltaEta": HistConf( [Axis(field="dibjet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{bb}$")] ),
        "dibjet_deltaEta_zoom" : HistConf( [Axis(field="dibjet_deltaEta", bins=25, start=0, stop=4, label=r"$\Delta \eta_{bb}$")] ),

        "dibjet_pt_max" : HistConf( [Axis(field="dibjet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{b1}$ [GeV]")] ),
        "dibjet_pt_max_zoom" : HistConf( [Axis(field="dibjet_pt_max", bins=50, start=0, stop=400, label=r"$p_T{b1}$ [GeV]")] ),

        "dibjet_pt_min" : HistConf( [Axis(field="dibjet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{b2}$ [GeV]")] ),
        "dibjet_pt_min_zoom" : HistConf( [Axis(field="dibjet_pt_min", bins=50, start=0, stop=400, label=r"$p_T{b2}$ [GeV]")] ),

        "dibjet_mass_max" : HistConf( [Axis(field="dibjet_mass_max", bins=100, start=0, stop=600, label=r"$M_{b1}$ [GeV]")] ),
        "dibjet_mass_max_zoom" : HistConf( [Axis(field="dibjet_mass_max", bins=50, start=0, stop=600, label=r"$M_{b1}$ [GeV]")] ),

        "dibjet_mass_min" : HistConf( [Axis(field="dibjet_mass_min", bins=100, start=0, stop=600, label=r"$M_{b2}$ [GeV]")] ),
        "dibjet_mass_min_zoom" : HistConf( [Axis(field="dibjet_mass_min", bins=50, start=0, stop=600, label=r"$M_{b2}$ [GeV]")] ),

        
#         "BDT_coarse": HistConf( [Axis(field="BDT_Hbb", bins=24, start=0, stop=1, label="BDT")],
#                          only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
#                                            ]),
        
#         "BDT": HistConf( [Axis(field="BDT_Hbb", bins=1000, start=0, stop=1, label="BDT")],
#                          only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
#                                            ]),
        
#         "GNN": HistConf( [Axis(field="GNN", bins=1000, start=0, stop=1, label="GNN")],
#                          only_categories = ['SR_mm_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ',
#                                             'SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm']),
        
#         "GNN_coarse": HistConf( [Axis(field="GNN", bins=24, start=0, stop=1, label="GNN")],
#                          only_categories = ['SR_mm_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ',
#                                             'SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm']),
        
#         "DNN_coarse": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
#                          only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
#                                            ]),
        
#         "DNN": HistConf( [Axis(field="DNN", bins=1000, start=0, stop=1, label="DNN")],
#                          only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
#                                             'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
#                                             'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
#                                            ]),
                        
        
#         "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
#         "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
#         "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

#         
#                          
#         
        
        
#         # 2D histograms:
#         "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
#                                    type="variable",   label="N. Jets (good)"),
#                               Axis(coll="events", field="JetGood_Ht",
#                                    bins=[0,80,150,200,300,450,700],
#                                    type="variable",
#                                    label="Jets $H_T$ [GeV]")]),
        
#         "dphi_jj_dr_jj": HistConf([ Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$"),
#                                     Axis(field="dijet_deltaPhi", bins=50, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$")]),
    }
)
