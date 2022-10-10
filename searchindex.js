Search.setIndex({docnames:["CONTRIBUTING","api","benchmark","cell_model","cli","development_install","docker","electro_mechanical_coupling","electrophysiology_model","gui","index","install","install_docker","install_pip","mechanics_model","release_test","simple_demo"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["CONTRIBUTING.md","api.rst","benchmark.ipynb","cell_model.md","cli.md","development_install.md","docker.md","electro_mechanical_coupling.md","electrophysiology_model.md","gui.md","index.md","install.md","install_docker.md","install_pip.md","mechanics_model.md","release_test.md","simple_demo.md"],objects:{"":[[1,0,0,"-","simcardems"]],"simcardems.Config":[[1,2,1,"","PCL"],[1,2,1,"","T"],[1,3,1,"","as_dict"],[1,2,1,"","bnd_cond"],[1,2,1,"","cell_init_file"],[1,2,1,"","disease_state"],[1,2,1,"","drug_factors_file"],[1,2,1,"","dt"],[1,2,1,"","dx"],[1,2,1,"","ep_ode_scheme"],[1,2,1,"","ep_preconditioner"],[1,2,1,"","ep_theta"],[1,2,1,"","fix_right_plane"],[1,2,1,"","hpc"],[1,2,1,"","linear_mechanics_solver"],[1,2,1,"","load_state"],[1,2,1,"","loglevel"],[1,2,1,"","lx"],[1,2,1,"","ly"],[1,2,1,"","lz"],[1,2,1,"","mechanics_ode_scheme"],[1,2,1,"","mechanics_use_continuation"],[1,2,1,"","mechanics_use_custom_newton_solver"],[1,2,1,"","num_refinements"],[1,2,1,"","outdir"],[1,2,1,"","popu_factors_file"],[1,2,1,"","pre_stretch"],[1,2,1,"","save_freq"],[1,2,1,"","set_material"],[1,2,1,"","spring"],[1,2,1,"","traction"]],"simcardems.DataCollector":[[1,4,1,"","names"],[1,3,1,"","register"],[1,4,1,"","results_file"],[1,3,1,"","store"]],"simcardems.DataLoader":[[1,3,1,"","get"],[1,4,1,"","size"]],"simcardems.EMCoupling":[[1,3,1,"","coupling_to_ep"],[1,3,1,"","coupling_to_mechanics"],[1,4,1,"","ep_mesh"],[1,3,1,"","ep_to_coupling"],[1,4,1,"","mech_mesh"],[1,3,1,"","mechanics_to_coupling"],[1,3,1,"","register_ep_model"],[1,3,1,"","register_mech_model"]],"simcardems.LandModel":[[1,4,1,"","As"],[1,4,1,"","Aw"],[1,4,1,"","Ta"],[1,3,1,"","Wactive"],[1,4,1,"","Zetas"],[1,4,1,"","Zetaw"],[1,4,1,"","cs"],[1,4,1,"","cw"],[1,4,1,"","dLambda"],[1,4,1,"","dt"],[1,3,1,"","start_time"],[1,4,1,"","t"],[1,3,1,"","update_Zetas"],[1,3,1,"","update_Zetaw"],[1,3,1,"","update_prev"],[1,3,1,"","update_time"]],"simcardems.MechanicsNewtonSolver":[[1,3,1,"","check_overloads_called"],[1,3,1,"","converged"],[1,3,1,"","default_solver_parameters"],[1,3,1,"","solve"],[1,3,1,"","solver_setup"]],"simcardems.MechanicsNewtonSolver_ODE":[[1,3,1,"","update_solution"]],"simcardems.MechanicsProblem":[[1,2,1,"","boundary_condition"],[1,3,1,"","solve"],[1,3,1,"","update_lmbda_prev"]],"simcardems.ORdmm_Land":[[1,5,1,"","Max"],[1,5,1,"","Min"],[1,1,1,"","ORdmm_Land"],[1,5,1,"","vs_functions_to_dict"]],"simcardems.ORdmm_Land.ORdmm_Land":[[1,3,1,"","F"],[1,3,1,"","I"],[1,3,1,"","default_initial_conditions"],[1,3,1,"","default_parameters"],[1,3,1,"","num_states"]],"simcardems.RigidMotionProblem":[[1,2,1,"","boundary_condition"],[1,3,1,"","rigid_motion_term"]],"simcardems.Runner":[[1,3,1,"","create_time_stepper"],[1,4,1,"","dt_mechanics"],[1,3,1,"","from_models"],[1,4,1,"","outdir"],[1,3,1,"","solve"],[1,3,1,"","store"],[1,4,1,"","t"],[1,4,1,"","t0"]],"simcardems.TimeStepper":[[1,4,1,"","T"],[1,4,1,"","dt"],[1,3,1,"","ms2ns"],[1,3,1,"","ns2ms"],[1,3,1,"","reset"],[1,4,1,"","total_steps"]],"simcardems.cli":[[1,5,1,"","main"]],"simcardems.config":[[1,1,1,"","Config"],[1,5,1,"","default_parameters"]],"simcardems.config.Config":[[1,2,1,"","PCL"],[1,2,1,"","T"],[1,3,1,"","as_dict"],[1,2,1,"","bnd_cond"],[1,2,1,"","cell_init_file"],[1,2,1,"","disease_state"],[1,2,1,"","drug_factors_file"],[1,2,1,"","dt"],[1,2,1,"","dx"],[1,2,1,"","ep_ode_scheme"],[1,2,1,"","ep_preconditioner"],[1,2,1,"","ep_theta"],[1,2,1,"","fix_right_plane"],[1,2,1,"","hpc"],[1,2,1,"","linear_mechanics_solver"],[1,2,1,"","load_state"],[1,2,1,"","loglevel"],[1,2,1,"","lx"],[1,2,1,"","ly"],[1,2,1,"","lz"],[1,2,1,"","mechanics_ode_scheme"],[1,2,1,"","mechanics_use_continuation"],[1,2,1,"","mechanics_use_custom_newton_solver"],[1,2,1,"","num_refinements"],[1,2,1,"","outdir"],[1,2,1,"","popu_factors_file"],[1,2,1,"","pre_stretch"],[1,2,1,"","save_freq"],[1,2,1,"","set_material"],[1,2,1,"","spring"],[1,2,1,"","traction"]],"simcardems.datacollector":[[1,1,1,"","DataCollector"],[1,1,1,"","DataGroups"],[1,1,1,"","DataLoader"]],"simcardems.datacollector.DataCollector":[[1,4,1,"","names"],[1,3,1,"","register"],[1,4,1,"","results_file"],[1,3,1,"","store"]],"simcardems.datacollector.DataGroups":[[1,2,1,"","ep"],[1,2,1,"","mechanics"]],"simcardems.datacollector.DataLoader":[[1,3,1,"","get"],[1,4,1,"","size"]],"simcardems.em_model":[[1,1,1,"","EMCoupling"]],"simcardems.em_model.EMCoupling":[[1,3,1,"","coupling_to_ep"],[1,3,1,"","coupling_to_mechanics"],[1,4,1,"","ep_mesh"],[1,3,1,"","ep_to_coupling"],[1,4,1,"","mech_mesh"],[1,3,1,"","mechanics_to_coupling"],[1,3,1,"","register_ep_model"],[1,3,1,"","register_mech_model"]],"simcardems.ep_model":[[1,5,1,"","define_conductivity_tensor"],[1,5,1,"","file_exist"],[1,5,1,"","handle_cell_inits"],[1,5,1,"","handle_cell_params"],[1,5,1,"","load_json"],[1,5,1,"","setup_ep_model"],[1,5,1,"","setup_splitting_solver_parameters"]],"simcardems.geometry":[[1,1,1,"","BaseGeometry"],[1,1,1,"","Geometry"],[1,1,1,"","SlabGeometry"],[1,5,1,"","create_boxmesh"],[1,5,1,"","refine_mesh"]],"simcardems.geometry.BaseGeometry":[[1,4,1,"","ep_mesh"],[1,2,1,"","mechanics_mesh"],[1,2,1,"","num_refinements"],[1,4,1,"","parameters"]],"simcardems.geometry.Geometry":[[1,3,1,"","parameters"]],"simcardems.geometry.SlabGeometry":[[1,4,1,"","parameters"]],"simcardems.land_model":[[1,1,1,"","LandModel"],[1,1,1,"","Scheme"]],"simcardems.land_model.LandModel":[[1,4,1,"","As"],[1,4,1,"","Aw"],[1,4,1,"","Ta"],[1,3,1,"","Wactive"],[1,4,1,"","Zetas"],[1,4,1,"","Zetaw"],[1,4,1,"","cs"],[1,4,1,"","cw"],[1,4,1,"","dLambda"],[1,4,1,"","dt"],[1,3,1,"","start_time"],[1,4,1,"","t"],[1,3,1,"","update_Zetas"],[1,3,1,"","update_Zetaw"],[1,3,1,"","update_prev"],[1,3,1,"","update_time"]],"simcardems.land_model.Scheme":[[1,2,1,"","analytic"],[1,2,1,"","bd"],[1,2,1,"","fd"]],"simcardems.mechanics_model":[[1,1,1,"","BoundaryConditions"],[1,1,1,"","ContinuationBasedMechanicsProblem"],[1,1,1,"","MechanicsProblem"],[1,1,1,"","RigidMotionProblem"],[1,5,1,"","float_to_constant"],[1,5,1,"","setup_diriclet_bc"],[1,5,1,"","setup_microstructure"]],"simcardems.mechanics_model.BoundaryConditions":[[1,2,1,"","dirichlet"],[1,2,1,"","rigid"]],"simcardems.mechanics_model.ContinuationBasedMechanicsProblem":[[1,3,1,"","solve"],[1,3,1,"","solve_for_control"]],"simcardems.mechanics_model.MechanicsProblem":[[1,2,1,"","boundary_condition"],[1,3,1,"","solve"],[1,3,1,"","update_lmbda_prev"]],"simcardems.mechanics_model.RigidMotionProblem":[[1,2,1,"","boundary_condition"],[1,3,1,"","rigid_motion_term"]],"simcardems.newton_solver":[[1,1,1,"","MechanicsNewtonSolver"],[1,1,1,"","MechanicsNewtonSolver_ODE"]],"simcardems.newton_solver.MechanicsNewtonSolver":[[1,3,1,"","check_overloads_called"],[1,3,1,"","converged"],[1,3,1,"","default_solver_parameters"],[1,3,1,"","solve"],[1,3,1,"","solver_setup"]],"simcardems.newton_solver.MechanicsNewtonSolver_ODE":[[1,3,1,"","update_solution"]],"simcardems.postprocess":[[1,1,1,"","Boundary"],[1,1,1,"","BoundaryNodes"],[1,5,1,"","center_func"],[1,5,1,"","extract_biomarkers"],[1,5,1,"","extract_last_beat"],[1,5,1,"","extract_traces"],[1,5,1,"","find_decaytime"],[1,5,1,"","find_duration"],[1,5,1,"","find_ttp"],[1,5,1,"","get_biomarkers"],[1,5,1,"","json_serial"],[1,5,1,"","load_data"],[1,5,1,"","load_mesh"],[1,5,1,"","load_times"],[1,5,1,"","make_xdmffiles"],[1,5,1,"","numpyfy"],[1,5,1,"","plot_peaks"],[1,5,1,"","plot_population"],[1,5,1,"","plot_state_traces"],[1,5,1,"","save_popu_json"],[1,5,1,"","stats"]],"simcardems.postprocess.Boundary":[[1,4,1,"","boundaries"],[1,4,1,"","center"],[1,4,1,"","node_A"],[1,4,1,"","node_B"],[1,4,1,"","node_C"],[1,4,1,"","node_D"],[1,4,1,"","node_E"],[1,4,1,"","node_F"],[1,4,1,"","node_G"],[1,4,1,"","node_H"],[1,3,1,"","nodes"],[1,4,1,"","xmax"],[1,4,1,"","xmin"],[1,4,1,"","ymax"],[1,4,1,"","ymin"],[1,4,1,"","zmax"],[1,4,1,"","zmin"]],"simcardems.postprocess.BoundaryNodes":[[1,2,1,"","center"],[1,2,1,"","node_A"],[1,2,1,"","node_B"],[1,2,1,"","node_C"],[1,2,1,"","node_D"],[1,2,1,"","node_E"],[1,2,1,"","node_F"],[1,2,1,"","node_G"],[1,2,1,"","node_H"],[1,2,1,"","xmax"],[1,2,1,"","xmin"],[1,2,1,"","ymax"],[1,2,1,"","ymin"],[1,2,1,"","zmax"],[1,2,1,"","zmin"]],"simcardems.save_load_functions":[[1,5,1,"","check_file_exists"],[1,5,1,"","decode"],[1,5,1,"","dict_to_h5"],[1,5,1,"","group_in_file"],[1,5,1,"","h5_to_dict"],[1,5,1,"","h5pyfile"],[1,5,1,"","load_cell_params_from_h5"],[1,5,1,"","load_initial_conditions_from_h5"],[1,5,1,"","load_state"],[1,5,1,"","save_cell_params_to_h5"],[1,5,1,"","save_cell_params_to_xml"],[1,5,1,"","save_state"],[1,5,1,"","save_state_variables_to_h5"],[1,5,1,"","save_state_variables_to_xml"]],"simcardems.setup_models":[[1,1,1,"","EMState"],[1,1,1,"","Runner"],[1,1,1,"","TimeStepper"],[1,5,1,"","create_progressbar"],[1,5,1,"","setup_EM_model"],[1,5,1,"","setup_ep_solver"],[1,5,1,"","setup_mechanics_solver"]],"simcardems.setup_models.EMState":[[1,2,1,"","coupling"],[1,2,1,"","mech_heart"],[1,2,1,"","solver"],[1,2,1,"","t0"]],"simcardems.setup_models.Runner":[[1,3,1,"","create_time_stepper"],[1,4,1,"","dt_mechanics"],[1,3,1,"","from_models"],[1,4,1,"","outdir"],[1,3,1,"","solve"],[1,3,1,"","store"],[1,4,1,"","t"],[1,4,1,"","t0"]],"simcardems.setup_models.TimeStepper":[[1,4,1,"","T"],[1,4,1,"","dt"],[1,3,1,"","ms2ns"],[1,3,1,"","ns2ms"],[1,3,1,"","reset"],[1,4,1,"","total_steps"]],"simcardems.utils":[[1,1,1,"","MPIFilt"],[1,5,1,"","compute_norm"],[1,5,1,"","enum2str"],[1,5,1,"","getLogger"],[1,5,1,"","local_project"],[1,5,1,"","print_mesh_info"],[1,5,1,"","remove_file"],[1,5,1,"","setup_assigner"],[1,5,1,"","sub_function"]],"simcardems.utils.MPIFilt":[[1,3,1,"","filter"]],simcardems:[[1,1,1,"","Config"],[1,1,1,"","DataCollector"],[1,1,1,"","DataLoader"],[1,1,1,"","EMCoupling"],[1,1,1,"","LandModel"],[1,1,1,"","MechanicsNewtonSolver"],[1,1,1,"","MechanicsNewtonSolver_ODE"],[1,1,1,"","MechanicsProblem"],[1,0,0,"-","ORdmm_Land"],[1,1,1,"","RigidMotionProblem"],[1,1,1,"","Runner"],[1,1,1,"","TimeStepper"],[1,0,0,"-","cli"],[1,0,0,"-","config"],[1,0,0,"-","datacollector"],[1,5,1,"","default_parameters"],[1,0,0,"-","em_model"],[1,0,0,"-","ep_model"],[1,0,0,"-","geometry"],[1,0,0,"-","land_model"],[1,0,0,"-","mechanics_model"],[1,0,0,"-","newton_solver"],[1,0,0,"-","postprocess"],[1,0,0,"-","save_load_functions"],[1,0,0,"-","setup_models"],[1,0,0,"-","utils"],[1,0,0,"-","version"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:property","5":"py:function"},terms:{"0":[1,2,3,4,14,15,16],"00":[2,3],"000000e":2,"001000e":2,"01":3,"010000e":2,"015389":2,"02":[1,3,4,15],"03":3,"04":[2,3],"05":[1,2,3,16],"06":2,"07":[2,3],"08":[2,3],"09":2,"0c195ff":2,"1":[0,1,2,3,10,14,15,16],"10":[2,3,4,15],"100":[3,4],"1000":[1,2,4,6,16],"100000e":2,"102":3,"106":3,"11":3,"110":3,"113":3,"118":3,"12":[2,3],"120":3,"123":3,"125":3,"13":3,"133":3,"138":2,"14":[2,3,7],"144":3,"146":3,"15":3,"150":3,"16":[2,3],"169":3,"17":[3,7],"170":3,"176":3,"18":3,"183":3,"19":[2,3],"1902":14,"193":3,"194":3,"1e":1,"2":[1,2,3,4,7,10,14,15,16],"20":[1,2,4,16],"200":[3,15],"2000":14,"2004":14,"2009":14,"2011":[3,10],"2014":7,"2017":3,"2022":[2,10],"21":3,"22":3,"226":3,"227464":2,"23":3,"230901":2,"238":3,"24":3,"242":3,"249432":2,"25":[2,3],"250":3,"26":3,"260":3,"262":3,"266":3,"27":3,"276":3,"28":[2,3],"288":3,"29":3,"292":3,"2j":14,"3":[1,2,3,7,14,15,16],"30":4,"300":3,"31":3,"32":2,"323":3,"33":3,"333":3,"34":3,"340":3,"3445":14,"3475":14,"35":3,"350":3,"3500":2,"355":3,"357":3,"36":3,"361561":2,"367":14,"37":3,"375":3,"386":3,"39":3,"3d":7,"4":[0,2,3,15],"400":3,"41":[2,3],"43":3,"430":3,"435":3,"438":3,"45":3,"46":3,"460":3,"470":3,"480":3,"49":3,"491":3,"493":3,"4_":14,"4_f":14,"4cc3e7a":2,"4d3a653":2,"5":[1,2,3,16],"50":3,"500":3,"501":14,"51":2,"516":3,"518":3,"52":3,"522":14,"53":3,"545":3,"55":[2,3],"552":3,"56":3,"562":3,"57":3,"592":3,"6":[2,3,7],"600":3,"603":3,"604":7,"609":3,"61":3,"615":7,"629":3,"63":3,"630ed09":2,"64":3,"648":3,"65":3,"661":3,"675":3,"68":3,"680":3,"7":[1,2,3,16],"700":3,"708":3,"711879":2,"72":3,"74":3,"75":3,"750":3,"76":3,"78":3,"8":3,"800":3,"805":3,"81":3,"82":3,"821":3,"83":3,"830650":2,"85":14,"850":3,"8501":[6,9],"86":3,"865697":2,"87":3,"870":3,"8_":14,"9":[2,3],"900":3,"92":3,"920":3,"93":3,"94":3,"99":3,"\u00e1":3,"\u00f3":3,"abstract":1,"c\u00e9cile":10,"case":[3,7,14,15,16],"class":[1,15],"default":[1,4,9],"do":[0,1,3,6,12,15],"enum":1,"final":16,"float":[1,4],"function":[1,14,15],"import":[2,7,14,15,16],"int":1,"new":[4,6,15],"public":0,"return":[1,15],"static":1,"super":15,"true":[1,15],"try":[9,13],"while":[1,7,14],A:[1,3,7,14],And:[15,16],As:[0,1,3],Being:0,By:9,For:[4,6,7,14],If:[0,1,4,6,9,13,15],In:[0,7,9,15,16],It:2,One:14,The:[0,1,7,9,10,11,12,14,15,16],There:7,These:[9,12],To:[0,6,7,9,14],_1:14,_2:14,_:14,__file__:2,__init__:15,__main__:15,__name__:15,_a:14,_p:14,_post_mechanics_solv:15,_print_messag:15,_t_releas:15,_time_stepp:15,a_:[3,14],a_f:14,ab:14,abc:1,abl:[1,14],about:[7,9,12,14],absolut:2,abus:0,acamk:3,acap:3,accept:0,account:0,act:0,action:[0,3],activ:[1,7,15,16],active_model:1,activemodel:1,adapt:0,add:15,add_trac:2,addit:[0,1],address:0,advanc:0,af:3,afca:3,afcaf:3,aff:3,after:[6,15],ag:0,ageo:3,ah:3,ahead:6,ahf:3,ai:3,aif:3,aka:6,alexand:14,alia:1,align:[0,14,15],all:[0,4,9,12],all_hash:2,allo_:3,allow:0,along:14,alreadi:1,also:[0,4,5,7,9,15,16],altern:[12,13],alwai:14,amd64:12,among:1,amount:1,amp:3,an:[0,1,9,12,15],analyt:[1,2,16],anca:3,andr:3,andrew:7,angular:14,ani:[0,1,7,10,14],anymor:6,ap:3,api:[10,16],app:6,appear:0,append:2,appl:13,appli:[0,1,15],appoint:0,approach:14,appropri:[0,1],ar:[0,2,6,7,10,12,13,14,16],architectur:12,arg0:1,arg1:1,arg2:1,arg3:1,arg4:1,arg:[1,4,15],argument:[4,15],arm64:12,arrhythmia:14,as_dict:[1,16],ask:0,ass:3,assert:15,assp:3,attack:0,attent:0,autogener:1,automat:[3,9],avail:0,aw:[1,3],axr:3,axrf:3,b:[1,3,14],b_:[3,14],b_f:14,back:7,background:6,ban:0,bar:4,base:[1,3,9,10,14],basegeometri:1,bcai:3,bcajsr:3,bcamk:3,bcass:3,bd:1,becaus:[2,6],been:6,befor:[0,13],begin:[3,14,15],behavior:0,benchmark:[1,10],benchmark_fold:2,best:0,better:1,between:15,biologi:[3,14],biomark:2,biomechan:7,biomed:7,biophys:14,bnd:1,bnd_cond:[1,4,16],bodi:0,bool:1,both:[0,7,12,14],boundari:[1,4,15],boundary_condit:1,boundarycondit:[1,16],boundarynod:1,box:1,br:2,branch:0,browser:9,bslmax:3,bsrmax:3,bt:3,btp:3,build:13,built:12,button:9,c7df02b:2,c:[2,3,14],c_m:1,ca:[1,15,16],cai:3,cajsr:3,calcium:[15,16],calib:3,call:[4,7,9,14,15,16],camka:3,camkb:3,camko:3,camkt:3,can:[0,2,4,6,7,9,10,12,13,14,15,16],cansr:3,cao:3,cardiac:[1,3,7,14],cardiaccellmodel:1,cardiolog:3,cardiomyocit:14,cardiomyocyt:3,cass:3,cat_:3,catrpn:3,catrpn_:3,catti:10,cauchi:14,cbcbeat:[1,8,10],cd:[3,13],cdot:14,cecil:10,cell:[1,7],cell_init:1,cell_init_fil:[1,4,16],cell_param:1,cellmodel:1,celltyp:3,cellular:[7,10,14],center:[1,2,15,16],center_func:1,cf7cc9d:2,chang:[0,2,9,15],channel:7,character:14,check_file_exist:1,check_overloads_cal:1,checkbox:0,checkout:[0,2],chi:1,chip:13,choic:[12,14],choos:4,ci:5,circumst:0,clarifi:0,classmethod:1,click:[0,9],clone:13,cmdnmax:3,coarser:7,code:[2,5,13],col:2,collabor:0,collect:2,column:2,com:[2,13],come:[5,14],command:[5,9,10,12,13],comment:0,commit:[0,2,5],common:[7,14],commun:0,compar:1,complaint:0,complet:1,compon:7,compress:1,comput:[3,6,7,9],computationalphysiolog:[2,6,9,12,13],compute_norm:1,concentr:[15,16],conda:13,condit:[1,4,15],confidenti:0,config:[15,16],configur:[14,16],conflict:0,consequ:7,conserv:14,consid:0,constant:[1,15],constitut:14,construct:0,consult:9,contact:0,contain:[1,4,7,9,12,16],content:4,continu:[1,2],continuationbasedmechanicsproblem:1,contract:[3,7,14],contribut:[10,14],control:[1,15],converg:[1,7],convers:0,convert:[1,15],copyright:10,core:10,correct:0,correctli:[2,15],could:0,coupl:[1,10,14,15],coupling_to_ep:1,coupling_to_mechan:1,coven:0,coverag:0,cpp:1,cpu:13,creat:[0,4,9,12,15,16],create_boxmesh:1,create_progressbar:1,create_time_stepp:1,cristob:3,critic:0,cs:[1,3],csqnmax:3,cube:1,current:[1,16],custom:16,cw:[1,3],d:[1,2,3,7],da:3,daemon:6,dap:3,data:[1,2],datafram:2,datagroup:1,dataload:1,date:2,datetim:2,daversin:10,dcai:3,dcajsr:3,dcamkt:3,dcansr:3,dcass:3,dcatrpn:3,dcd:3,dd:3,debug:4,declar:14,decod:1,decompos:14,deem:[0,1],def:15,default_initial_condit:1,default_paramet:1,default_solver_paramet:1,defaultdict:2,defin:[0,7,14],define_conductivity_tensor:1,deform:14,delet:9,delta:[1,3],delta_:3,densiti:14,depend:[7,10,11,15],deriv:14,derogatori:0,describ:7,det:14,detail:[0,2,14],determin:[0,1],dev:5,develop:[3,9,10,11,13],df:[2,3],dfca:3,dfcaf:3,dfcafp:3,dff:3,dffp:3,dh:3,dhf:3,dhl:3,dhlp:3,dhsp:3,di:3,dict:1,dict_to_h5:1,dictionari:1,dif:3,differ:[0,1,2,9,16],difficult:11,difp:3,direct:[1,4,14],directli:[0,13],directori:[4,9,16],dirichlet:[1,4,15,16],disabl:0,discret:[4,7],discretis:7,discuss:[0,7],disease_st:[1,16],disp:3,displac:[1,14,15,16],displai:2,dit:6,dj:3,djca:3,djp:3,djrelnp:3,djrelp:3,dki:3,dkss:3,dlambda:[1,3],dm:3,dml:3,dnai:3,dnass:3,dnca:3,docker:[10,11,13],dockerfil:12,document:[0,3],documentaion:10,doe:[1,2],doesn:6,dolfin:[1,15],don:[0,6],done:[6,9],drop:[15,16],drug:4,drug_factors_fil:[1,4,16],dss:3,dt:[1,3,4,16],dt_mechan:1,dti_:3,dtmb:3,durat:3,dure:15,dv:[1,3],dx:[1,3,4,16],dxk_:3,dxr:3,dxrf:3,dxs_:3,dxw:3,dzeta:3,dzetaw:3,e1002061:3,e1:3,e:[0,1,2,3,4,5,7,9,14,15],e_:3,each:[0,7,9,16],edit:[0,5],either:[0,1,9,13],ek:3,electr:7,electromechan:14,electron:0,electrophysiolog:[7,10,15],element:14,email:0,emcoupl:[1,3],empathi:0,emploi:14,empti:1,emstat:1,en:8,ena:3,enabl:0,encompass:7,end:[3,4,14,15],energi:[1,14],enforc:14,engin:[7,14],ensur:7,enum2str:1,enumcl:1,enumer:[1,2],environ:[0,12],ep:[1,3,4,9],ep_mesh:1,ep_ode_schem:[1,16],ep_precondition:[1,16],ep_solv:1,ep_theta:[1,16],ep_to_coupl:1,epi:3,equat:[2,7,14],equilibrium:14,esac_:3,eta:[1,3],etal:3,ethnic:0,evalu:2,event:0,everyon:0,exampl:[0,4,6],except:1,excit:14,exec:6,execut:[4,9,12],exist:[1,2,4],exit:4,expect:0,experi:0,experiment:3,explicit:0,express:0,extern:[1,14],extra:[9,15],extract_biomark:1,extract_last_beat:1,extract_trac:1,f0:1,f63afc9:2,f:[1,2,3,14,15],f_:3,face:0,factor:4,fail:15,failur:1,fair:0,faith:0,fallingfactori:3,fals:[1,2,15,16],fca:3,fcaf:3,fcafp:3,fcap:3,fcass:3,fd:[1,3],feedback:7,fenic:[10,11],fenics_plotli:9,ff:3,ffp:3,fiber:14,ficalp:3,field:[1,15],fig:2,figur:[15,16],file:[1,4,9,16],file_exist:1,filemod:1,filenam:1,fill:0,filter:1,finalp:3,finap:3,find:0,find_decaytim:1,find_dur:1,find_ttp:1,finish:0,finit:14,finsberg:10,first:[0,6,9,14,15],fitop:3,fix:[1,15],fix_right_plan:[1,16],fjrelp:3,fjupp:3,float_to_const:1,flow:[0,15],fmax:1,fmin:1,fname:1,focus:0,folder:[9,12],follow:[0,4,5,6,9,12,16],forc:[1,15],forget:0,fork:0,formatt:5,formul:3,forward:9,foster:0,found:10,fp:3,frac:[3,14],framework:14,free:0,frequenc:4,from:[0,1,2,3,7,9,12,13,14,15,16],from_model:1,fs:[3,14],fss:3,further:[0,7],futhermor:7,g:[3,4],gamma:3,gamma_0:15,gamma_1:15,gammasu:3,gammaw:3,gammawu:3,gender:0,gener:[2,3],genericmatrix:1,genericvector:1,geometri:[7,15,16],geq:15,gerhard:14,get:[0,1,9,12],get_biomark:1,getlogg:1,ghcr:[6,9,12],git:[0,2,13],git_commit_hash:2,git_hash:2,github:[0,2,10,13],given:14,gk:3,gk_:3,gkb:3,gkr:3,gna:3,gnal:3,gncx:3,gnu:10,go:[2,9],good:[0,9,12],gotran:[1,3],gpca:3,gracefulli:0,gradient:14,graph_object:2,graphic:10,greater:15,green:14,grl1:[1,16],group:1,group_in_fil:1,gsac_:3,gto:3,guccion:4,guess:1,h5:[4,15,16],h5_filenam:1,h5_to_dict:1,h5group:1,h5name:1,h5pyfil:1,h:[2,3],h_:3,ha:[6,7],half:15,hand:1,handle_cell_init:1,handle_cell_param:1,happen:7,hara:3,harald:7,harass:0,hard:2,harm:0,hash:[2,9],have:[0,6,7,13,15],hca:3,healthi:[1,16],heart:[1,14],height:2,help:[0,4,6],henrik:10,henriknf:[0,10],herck:10,here:[0,2,9,14],hf:[1,3],high:7,hl:3,hlp:3,hlss:3,hlssp:3,hna:3,ho09:14,hol00:14,holohan:3,holzapfel:14,holzapfelogden:4,home:9,hood:14,hook:5,hovertempl:2,how:[2,4,6,9,12],howev:2,hp:3,hpc:[1,4,16],hs:3,hsp:3,hss:3,hssp:3,html:8,http:[0,2,8,9,13],hub:12,human:3,hyperelast:14,i:[1,2,3,7,9,14,15],iF:3,iS:3,i_1:14,i_:14,ic:4,icab:3,icak:3,icana:3,id:9,idea:[9,15],ident:0,ifp:3,ii:14,ik1:3,ik:3,ik_:3,ikb:3,ikr:3,ils:10,ilsbeth:10,imag:6,imageri:0,impact:2,implement:[2,7,15],import_tim:2,improv:7,ina:3,inab:3,inaca_:3,inak:3,inal:3,inappropri:0,inc:14,incid:0,includ:0,inclus:0,incompress:14,increas:1,index:1,indic:4,individu:[0,9],inf:3,info:[4,14],inform:[0,7,8],infp:3,init_condit:1,initi:[1,4],input:9,insid:9,instal:[4,6,9],instanc:0,instant:[15,16],instruct:12,insult:0,integ:4,integr:7,intel:13,interact:0,intercellular:[15,16],interest:0,interfac:10,intern:15,interpol:7,inv_lmbda:1,invari:14,investig:0,involv:7,io:[6,8,9,12],ip:3,ipca:3,is_fil:2,isac:3,isac_:3,isinst:15,isol:12,isp:3,iss:3,issu:0,istim:3,item:2,iter:15,iterdir:2,ito:3,its:0,itself:14,j:[3,14],j_:3,jca:3,jdiff:3,jdiffk:3,jdiffna:3,jin:3,jleak:3,jnakk:3,jnakna:3,jncxca_:3,jncxna_:3,joakim:7,john:14,join:2,joinpath:[15,16],jonathan:3,journal:3,jp:3,jrel:3,jrel_:3,jrelnp:3,jrelp:3,json:[2,4],json_seri:1,jss:3,jtr:3,jup:3,jupnp:3,jupp:3,k1m:3,k1p:3,k2m:3,k2n:3,k2p:3,k3m:3,k3p:3,k3p_:3,k3pp:3,k3pp_:3,k4m:3,k4p:3,k4p_:3,k4pp:3,k4pp_:3,k:3,k_:3,kasymm:3,kb:3,kcaoff:3,kcaon:3,keep:2,kei:2,kentish:3,keyerror:1,khp:3,ki:3,kirchhoff:14,kki:3,kko:3,km2n:3,kmbsl:3,kmbsr:3,kmcaact:3,kmcam:3,kmcamk:3,kmcmdn:3,kmcsqn:3,kmgatp:3,kmn:3,kmtrpn:3,kna_:3,knai:3,knai_:3,knao:3,knao_:3,knap:3,know:14,known:[0,11],ko:3,ksca:3,kss:3,ksu:3,ktrpn:3,ku:3,kuw:3,kw:3,kwarg:[1,15],kwu:3,kxkur:3,l:3,l_x:15,la:1,laboratori:10,lagrang:14,lambda:[3,14,15,16],lambda_:3,land:3,landmodel:1,languag:0,later:10,latest:[8,12,13],law:14,leadership:0,left:[1,3,14],legaci:[11,13],length:15,leq:3,level:[0,7,10],lgpl:10,librari:[2,16],lightweight:9,like:15,line:[9,10],linear:14,linear_mechanics_solv:[1,16],linear_solv:1,linearis:7,link:[0,2],linter:5,list:[1,2,9],lmbda:[1,3],lmbda_prev:1,load:[2,4,14],load_cell_params_from_h5:1,load_data:1,load_initial_conditions_from_h5:1,load_json:1,load_mesh:1,load_stat:[1,4,16],load_tim:1,loader:1,local_project:1,localhost:9,locat:9,log:[1,3,15],loglevel:[1,4,16],look:9,loss:7,lot:9,lower:0,lowest:1,lph:3,lx:[1,4,15,16],ly:[1,4,16],lz:[1,4,16],m:[3,4,5,6,9,13],machin:6,made:0,mai:[0,1],mail:0,main:[1,7,15],maintain:0,make:[0,2,6,14,15],make_subplot:2,make_xdmffil:[1,16],man:13,manual:9,mark:0,marker:1,markerfunct:1,martyn:14,materi:[1,4,14],mathbb:14,mathbf:[14,15],mathcal:3,mathemat:14,mathrm:14,matplotlib:2,max:[1,3],max_column:2,mcculloch:7,mean:1,measur:3,mech_heart:[1,15],mech_mesh:1,mechan:[1,4,9,15],mechanics_mesh:1,mechanics_ode_schem:[1,16],mechanics_to_coupl:1,mechanics_use_continu:[1,16],mechanics_use_custom_newton_solv:[1,16],mechanicsnewtonsolv:1,mechanicsnewtonsolver_od:1,mechanicsproblem:1,mechano:7,media:0,member:0,membran:[1,7,15,16],merg:0,mesh:[1,4,9],messag:[4,15],method:[0,7,14,15],mgadp:3,mgatp:3,middl:15,might:2,millisecond:[1,15],min087:3,min12:3,min:1,ml:3,mlss:3,mode:[3,6],model:[1,4,9,10,15],modifi:1,modul:[8,16],molecular:[3,14],momentum:14,monodomain:[7,8],monodomainsolv:8,more:[2,8,9,12,14],most:[15,16],motion:[1,14],move:1,mpifilt:1,ms2n:1,ms:2,mss:3,much:[4,7],multipli:14,mump:[1,16],myocardium:14,n0:1,n:[2,4],nabla:14,nai:3,name:[1,2,3,6,12],nanosecond:[1,15],nao:3,nash:14,nass:3,nation:0,natur:14,nca:3,necessari:[0,15,16],need:[1,6,7,9,13],neg:1,neumann:1,newton:1,newtonsolv:1,next:[15,16],nice:0,nicola:3,nieder:3,nl:1,node:[1,7],node_:1,node_a:1,node_b:1,node_c:1,node_d:1,node_f:1,node_g:1,node_h:1,non:14,none:[1,4,15,16],nonetyp:1,nonlinear:14,nonlinearproblem:1,note:[0,1,5,6,14,15],noth:1,novel:3,now:[6,15],np04:14,np:2,ns2m:[1,15],ns:3,ntm:3,ntrpn:3,num_model:1,num_refin:[1,4,16],num_stat:1,number:[1,4],numer:7,numpi:2,numpyfi:1,o:[3,4,15],obj:1,object:1,oblig:0,obtain:7,od:[1,7,15],off:[4,15],offens:0,offici:0,offlin:0,ogden:14,oharaviragvarror11:3,omega:14,onc:[0,4,6,13,15],one:[0,7,14,15],onli:15,onlin:0,open:[0,2,9,16],operatornam:3,opposit:15,option:[1,2,4,9,10],order:[2,6,7,9,15],ordereddict:1,ordmm_land_em_coupl:1,org:0,orient:0,origin:14,os:1,osn:7,other:[0,7,15],otherwis:[0,1,3,4],out:2,outdir:[1,4,15,16],output:[4,9,15,16],over:2,overview:0,owner:0,p:[3,6,9,14],p_:3,pace:1,packag:[5,6,9,11,13],page:[0,9],panda:2,panfilov:14,paper:14,param:1,paramet:[1,9],params_list:1,paraview:16,parent:[2,15],park:3,part:[10,15],partial:14,particip:0,pass:[0,7],path:[1,2,4,15,16],pathlib:[2,15,16],pathlik:1,pca:3,pcab:3,pcak:3,pcakp:3,pcana:3,pcanap:3,pcap:3,pcl:[1,16],pd:2,perman:0,permiss:0,person:0,phenomena:7,phi:3,phicak:3,phicana:3,philosoph:14,physic:[0,14],piola:14,pip:[5,9,10,11],pkna:3,place:1,plane:1,pleas:[0,6,9],plo:3,plot:[9,15,16],plot_peak:1,plot_popul:1,plot_state_trac:[1,15,16],plotli:2,plt:2,pnab:3,pnak:3,png:16,point:[1,16],polici:0,polit:0,popu_factors_fil:[1,4,16],popul:4,population_fold:1,port:9,posit:[0,1,15],posixpath:16,post:0,postprocess:[4,15,16],potenti:[1,3,7,15,16],pprint:16,pr:0,pre:[1,5,12,15],pre_stretch:[1,15,16],precis:7,precondition:1,prefer:[11,12],prima:3,princip:14,print:[2,4,15,16],print_mesh_info:1,privat:0,problem:[1,4],process:7,profession:0,progress:[4,14],project:[0,10,12],properti:[1,4],propos:0,provid:[1,4,12],psi:14,psi_a:14,psi_p:14,publish:0,pull:12,puls:[1,10,14,15],pure:11,purpos:0,pwd:6,pyplot:2,python3:[4,6,9,13],python:[4,5,11,16],qca:3,qna:3,quantiti:14,question:0,quit:9,r:[1,2,3],race:0,rad:3,radio:9,rai:14,rais:1,raise_on_fals:1,rapid:15,re:0,read:[0,12],reader:14,readi:0,readthedoc:8,reason:0,rebuild:6,recomput:7,record:1,recov:3,redistribut:1,reduct:15,reentrant:14,ref:3,refin:[1,4],refine_mesh:1,regard:0,regardless:0,regist:1,register_ep_model:1,register_mech_model:1,reject:0,rel:3,rel_max_displacement_perc_z:2,relat:14,releas:[10,16],release_test_result:15,releaserunn:15,relev:[7,14],religion:0,relp:3,remedio:3,remov:0,remove_fil:1,render:0,repercuss:0,repo:[5,12,13],repolaris:1,report:0,repositori:[0,13],repres:[0,1,7,14],represent:0,requir:[5,7,9],rerun:2,research:10,reset:1,reset_st:1,reset_tim:1,resolut:7,resolv:0,resourc:[0,6],respect:[0,14],result:[0,1,2,4,15,16],results_:9,results_fil:[1,2],results_simple_demo:16,retriev:1,return_interv:1,review:0,right:[0,1,3,14],rigid:[1,4],rigid_motion_term:1,rigidmotionproblem:1,rk_:3,rkr:3,rm:[6,9],root:12,row:2,royal:14,rs:3,rudi:3,run:[0,2,4,5,10,11,12,13],run_id:1,runner:[1,15,16],rw:3,s0:1,s:[0,1,3,7,14],same:[1,2],sander:3,satisfi:14,save:[4,16],save_cell_params_to_h5:1,save_cell_params_to_xml:1,save_freq:[1,4,16],save_popu_json:1,save_st:1,save_state_variables_to_h5:1,save_state_variables_to_xml:1,scale:4,scale_:3,scatter:2,scheme:[1,7,16],scienc:14,section:7,see:[0,1,2,4,8,15,16],select:9,self:[1,15],separ:[0,7,9],set:[0,1,4,5,15,16],set_log_level:15,set_materi:[1,4,16],setup:[1,9],setup_assign:1,setup_diriclet_bc:1,setup_em_model:1,setup_ep_model:1,setup_ep_solv:1,setup_mechanics_solv:1,setup_microstructur:1,setup_splitting_solver_paramet:1,sever:7,sexual:0,shape:1,sheet:14,should:[1,5,6,13],show:[0,2,4,16],showlegend:2,shown:16,side:[1,15],signific:7,silicon:13,simcardem:[2,4,6,9,10,11,12,15,16],simcardems_vers:2,simcardio:10,simpl:[2,10],simul:[2,3,4,7,15,16],simula:0,size:[0,1,4],slab:[2,14,15],slabgeometri:[1,15],slap:15,smith:3,so:[0,1,3,6,7,9,15],social:0,societi:14,solid:14,solut:[2,7],solv:[0,1,14,15,16],solve_for_control:1,solver:[1,7,14,15],solver_setup:1,some:[0,1,7,9,15],someth:9,son:14,sor:[1,16],sort_valu:2,sourc:[0,1,13],space:0,spatial:[4,7],speak:14,special:7,specif:[0,2,14,16],specifi:[1,4],split:14,splittingsolv:1,spring:[1,16],sqrt:[3,14],ss:3,st_progress:1,stabil:7,stamp:1,start:[0,16],start_tim:1,stat:1,state:[1,4,7,15,16],state_prev:1,state_trac:16,step:[1,4,6,7,16],steven:3,stiff:1,storag:9,store:[1,9,15],str:1,strain:14,streamlit:9,strength:7,stress:[1,14],stretch:[1,7,14,15,16],string:1,strongli:7,structur:14,studi:14,sub_funct:1,subclass:15,subfold:9,submit:0,subplot:2,subplot_titl:2,subsystem:13,suffix:1,suggest:0,sum:14,sundn:7,support:12,sure:[0,6,15],swap:2,swo:7,system:[1,5,15],szl:3,t0:1,t:[0,1,2,3,4,6,12,14,15,16],t_a:[14,15,16],t_releas:15,ta:[1,3],tab:[0,9],take:[0,6,9,15],taken:[2,14],tau_:3,taylor:14,td:3,team:0,templat:0,temporari:0,temporarili:0,tension:[3,7,14,15,16],tensor:14,test:[0,5,10],text:[2,3,4,15],tf:3,tfca:3,tfcaf:3,tfcafp:3,tff:3,tffp:3,th:3,than:0,thei:0,them:0,therefor:7,theta:1,thf:3,thi:[0,1,3,4,6,7,9,12,14,15,16],think:7,thl:3,thlp:3,thoma:3,thorvaldsen:7,threaten:0,three:7,threshold:1,through:[0,1,7,12,15],thsp:3,ti:3,tif:3,tifp:3,time:[1,2,3,4,7,9,15,16],time_point:1,time_stamp:1,time_stepp:1,time_to_max_displacement_z:2,timestamp:2,timestepp:1,tisp:3,tissu:[7,10],tj:3,tjca:3,tjp:3,tm:3,tmb:3,tml:3,tmp:3,to_datetim:2,tol:1,tom:7,tot_:3,total:14,total_dof:1,total_step:1,toward:0,tp:3,tr:14,trace:[2,15,16],track:2,traction:[1,16],transact:14,transmembran:1,tref:3,trigger:[15,16],troll:0,trpn:3,trpn_:3,trpnmax:3,ttot:3,tupl:1,turn:[4,15],tutori:0,two:7,txk_:3,txr:3,txrf:3,txs_:3,type:[1,4,15],u:[1,14,15,16],ubuntu:13,ui:0,unaccept:0,under:[9,10],underli:15,understand:0,undiseas:3,union:1,unit:14,unwelcom:0,up:[0,1,6,9,15],updat:[0,6],update_cb:1,update_layout:2,update_lmbda_prev:1,update_prev:1,update_solut:1,update_tim:1,update_zeta:1,update_zetaw:1,url:9,us:[0,1,3,4,5,8,9,10,11,12,13,14,16],usag:[4,16],use_custom_newton_solv:1,use_n:1,user:10,v:[1,2,3,6,14,15,16],valid:3,valu:[1,2,3,7,15],van:10,vari:15,variabl:[1,15,16],variat:1,varr:3,vcell:3,vector:14,ventricular:3,verifi:2,version:[0,2,4,10,11,13],vffrt:3,vfrt:3,via:[0,7],view:16,viewpoint:0,vir:3,virtual:6,visual:[9,16],vjsr:3,vmyo:3,vnsr:3,voltag:16,vs:1,vs_functions_to_dict:1,vss:3,w:[1,14],wa:1,wactiv:1,wai:[0,7,11,12,15],wall:7,want:[1,4,6,9,12,13,16],warn:[4,15],wca:3,we:[0,2,7,8,9,12,14,15,16],welcom:0,well:0,what:[0,14],when:[0,6,15],whenev:2,where:[1,2,9,14,15],which:[0,5,7,9,11,12,14,15,16],whiteout:7,who:0,width:2,wiki:0,wilei:14,wish:0,within:[0,12],without:[0,14],wna:3,wnaca:3,word:[7,15],work:3,workflow:0,would:15,wrap:1,x:[1,2,4,14,15],x_:3,x_prev:1,xdmf:16,xk1ss:3,xk_:3,xkb:3,xmax:1,xmin:1,xr:3,xrf:3,xrss:3,xs1ss:3,xs2ss:3,xs:[1,3],xs_:3,xu:3,xw:[1,3],xw_:3,y:[1,2,4,14],ymax:1,ymin:1,yoram:3,you:[0,1,2,4,5,6,9,12,13,16],your:[0,6,9,10],yourselv:12,z:[1,4],zca:3,zero:15,zeta:[1,3],zetas_prev:1,zetaw:[1,3],zetaw_prev:1,zip:2,zk:3,zmax:1,zmin:1,zna:3},titles:["Contributing","API documentaion","Benchmark","Cellular model","Command line interface","Development installation","Run using Docker","Electro-Mechanical coupling","Tissue level electrophysiology model","Graphical user interface","Simula Cardiac Electro-Mechanics Solver","Installation","Install with Docker","Install with <code class=\"docutils literal notranslate\"><span class=\"pre\">pip</span></code>","Tissue level mechanics model","Release test","Simple demo"],titleterms:{"new":0,activ:14,again:6,api:1,attribut:0,author:10,benchmark:2,between:7,build:12,cardiac:10,cellular:3,cli:1,code:[0,10],command:[4,6],commun:10,conduct:0,config:1,contain:6,content:[1,10],contribut:0,contributor:0,coupl:7,creat:6,datacollector:1,delet:6,demo:[10,16],develop:5,differ:7,discret:14,docker:[6,9,12],documentaion:1,electro:[7,10],electrophysiolog:8,em_model:1,enforc:0,ep:7,ep_model:1,execut:6,express:3,fenic:13,geometri:1,graphic:9,gui:9,guid:0,imag:12,instal:[5,10,11,12,13],interfac:[4,9],land_model:1,level:[8,14],licens:10,line:[4,6],linux:13,mac:13,mathemat:10,mechan:[7,10,14],mechanics_model:1,mesh:7,model:[3,7,8,14],modul:1,newton_solv:1,ordmm_land:1,ordmm_land_em_coupl:3,our:0,outsid:9,own:12,paramet:3,passiv:14,pip:13,pledg:0,postprocess:[1,9],process:0,programm:10,pull:0,refer:[3,7,10,14],releas:15,request:0,respons:0,result:9,run:[6,9],save_load_funct:1,scope:0,script:6,setup_model:1,simcardem:[1,13],simpl:16,simul:9,simula:10,softwar:14,solv:7,solver:10,sourc:10,standard:0,start:[6,9],state:3,stop:6,test:15,theori:10,tissu:[8,14],transfer:7,us:6,user:[9,13],util:1,variabl:7,verif:10,version:1,window:13,your:12}})