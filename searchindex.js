Search.setIndex({docnames:["CONTRIBUTING","api","benchmark","cell_model","cli","docker","electro_mechanical_coupling","electrophysiology_model","gui","index","install","install_docker","install_pip","mechanics_model","release_test","simple_demo","ventricular_geometry","visualization"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["CONTRIBUTING.md","api.rst","benchmark.ipynb","cell_model.md","cli.md","docker.md","electro_mechanical_coupling.md","electrophysiology_model.md","gui.md","index.md","install.md","install_docker.md","install_pip.md","mechanics_model.md","release_test.md","simple_demo.md","ventricular_geometry.md","visualization.md"],objects:{"":[[1,0,0,"-","simcardems"]],"simcardems.Config":[[1,2,1,"","PCL"],[1,2,1,"","T"],[1,3,1,"","as_dict"],[1,2,1,"","bnd_rigid"],[1,2,1,"","cell_init_file"],[1,2,1,"","disease_state"],[1,2,1,"","drug_factors_file"],[1,2,1,"","dt"],[1,2,1,"","ep_ode_scheme"],[1,2,1,"","ep_preconditioner"],[1,2,1,"","ep_theta"],[1,2,1,"","fix_right_plane"],[1,2,1,"","geometry_path"],[1,2,1,"","geometry_schema_path"],[1,2,1,"","linear_mechanics_solver"],[1,2,1,"","load_state"],[1,2,1,"","loglevel"],[1,2,1,"","mechanics_ode_scheme"],[1,2,1,"","mechanics_use_continuation"],[1,2,1,"","mechanics_use_custom_newton_solver"],[1,2,1,"","num_refinements"],[1,2,1,"","outdir"],[1,2,1,"","popu_factors_file"],[1,2,1,"","pre_stretch"],[1,2,1,"","save_freq"],[1,2,1,"","set_material"],[1,2,1,"","show_progress_bar"],[1,2,1,"","spring"],[1,2,1,"","traction"]],"simcardems.DataCollector":[[1,4,1,"","names"],[1,3,1,"","register"],[1,4,1,"","results_file"],[1,3,1,"","store"]],"simcardems.DataLoader":[[1,4,1,"","ep_mesh"],[1,3,1,"","extract_value"],[1,3,1,"","get"],[1,4,1,"","mech_mesh"],[1,4,1,"","size"]],"simcardems.EMCoupling":[[1,3,1,"","coupling_to_ep"],[1,3,1,"","coupling_to_mechanics"],[1,4,1,"","ep_mesh"],[1,3,1,"","ep_to_coupling"],[1,4,1,"","mech_mesh"],[1,3,1,"","mechanics_to_coupling"],[1,3,1,"","register_ep_model"],[1,3,1,"","register_mech_model"]],"simcardems.LandModel":[[1,4,1,"","As"],[1,4,1,"","Aw"],[1,4,1,"","Ta"],[1,3,1,"","Wactive"],[1,4,1,"","Zetas"],[1,4,1,"","Zetaw"],[1,4,1,"","cs"],[1,4,1,"","cw"],[1,4,1,"","dLambda"],[1,4,1,"","dt"],[1,3,1,"","start_time"],[1,4,1,"","t"],[1,3,1,"","update_Zetas"],[1,3,1,"","update_Zetaw"],[1,3,1,"","update_prev"],[1,3,1,"","update_time"]],"simcardems.MechanicsNewtonSolver":[[1,3,1,"","check_overloads_called"],[1,3,1,"","converged"],[1,3,1,"","default_solver_parameters"],[1,3,1,"","solve"],[1,3,1,"","solver_setup"]],"simcardems.MechanicsNewtonSolver_ODE":[[1,3,1,"","update_solution"]],"simcardems.MechanicsProblem":[[1,3,1,"","solve"],[1,4,1,"","u_subspace_index"],[1,3,1,"","update_lmbda_prev"]],"simcardems.ORdmm_Land":[[1,5,1,"","Max"],[1,5,1,"","Min"],[1,1,1,"","ORdmm_Land"],[1,5,1,"","vs_functions_to_dict"]],"simcardems.ORdmm_Land.ORdmm_Land":[[1,3,1,"","F"],[1,3,1,"","I"],[1,3,1,"","default_initial_conditions"],[1,3,1,"","default_parameters"],[1,3,1,"","num_states"]],"simcardems.RigidMotionProblem":[[1,3,1,"","rigid_motion_term"],[1,4,1,"","u_subspace_index"]],"simcardems.Runner":[[1,3,1,"","create_time_stepper"],[1,4,1,"","dt_mechanics"],[1,3,1,"","from_models"],[1,4,1,"","outdir"],[1,3,1,"","save_state"],[1,3,1,"","solve"],[1,3,1,"","store"],[1,4,1,"","t"],[1,4,1,"","t0"]],"simcardems.TimeStepper":[[1,4,1,"","T"],[1,4,1,"","dt"],[1,3,1,"","ms2ns"],[1,3,1,"","ns2ms"],[1,3,1,"","reset"],[1,4,1,"","total_steps"]],"simcardems.cli":[[1,5,1,"","main"]],"simcardems.config":[[1,1,1,"","Config"],[1,5,1,"","default_parameters"]],"simcardems.config.Config":[[1,2,1,"","PCL"],[1,2,1,"","T"],[1,3,1,"","as_dict"],[1,2,1,"","bnd_rigid"],[1,2,1,"","cell_init_file"],[1,2,1,"","disease_state"],[1,2,1,"","drug_factors_file"],[1,2,1,"","dt"],[1,2,1,"","ep_ode_scheme"],[1,2,1,"","ep_preconditioner"],[1,2,1,"","ep_theta"],[1,2,1,"","fix_right_plane"],[1,2,1,"","geometry_path"],[1,2,1,"","geometry_schema_path"],[1,2,1,"","linear_mechanics_solver"],[1,2,1,"","load_state"],[1,2,1,"","loglevel"],[1,2,1,"","mechanics_ode_scheme"],[1,2,1,"","mechanics_use_continuation"],[1,2,1,"","mechanics_use_custom_newton_solver"],[1,2,1,"","num_refinements"],[1,2,1,"","outdir"],[1,2,1,"","popu_factors_file"],[1,2,1,"","pre_stretch"],[1,2,1,"","save_freq"],[1,2,1,"","set_material"],[1,2,1,"","show_progress_bar"],[1,2,1,"","spring"],[1,2,1,"","traction"]],"simcardems.datacollector":[[1,1,1,"","DataCollector"],[1,1,1,"","DataGroups"],[1,1,1,"","DataLoader"]],"simcardems.datacollector.DataCollector":[[1,4,1,"","names"],[1,3,1,"","register"],[1,4,1,"","results_file"],[1,3,1,"","store"]],"simcardems.datacollector.DataGroups":[[1,2,1,"","ep"],[1,2,1,"","mechanics"]],"simcardems.datacollector.DataLoader":[[1,4,1,"","ep_mesh"],[1,3,1,"","extract_value"],[1,3,1,"","get"],[1,4,1,"","mech_mesh"],[1,4,1,"","size"]],"simcardems.em_model":[[1,1,1,"","EMCoupling"]],"simcardems.em_model.EMCoupling":[[1,3,1,"","coupling_to_ep"],[1,3,1,"","coupling_to_mechanics"],[1,4,1,"","ep_mesh"],[1,3,1,"","ep_to_coupling"],[1,4,1,"","mech_mesh"],[1,3,1,"","mechanics_to_coupling"],[1,3,1,"","register_ep_model"],[1,3,1,"","register_mech_model"]],"simcardems.ep_model":[[1,5,1,"","default_conductivities"],[1,5,1,"","default_microstructure"],[1,5,1,"","define_conductivity_tensor"],[1,5,1,"","file_exist"],[1,5,1,"","handle_cell_inits"],[1,5,1,"","handle_cell_params"],[1,5,1,"","load_json"],[1,5,1,"","setup_ep_model"],[1,5,1,"","setup_splitting_solver_parameters"]],"simcardems.geometry":[[1,1,1,"","BaseGeometry"],[1,1,1,"","LeftVentricularGeometry"],[1,1,1,"","SlabGeometry"],[1,5,1,"","create_boxmesh"],[1,5,1,"","create_slab_facet_function"],[1,5,1,"","create_slab_microstructure"],[1,5,1,"","load_geometry"],[1,5,1,"","refine_mesh"]],"simcardems.geometry.BaseGeometry":[[1,3,1,"","default_markers"],[1,3,1,"","default_parameters"],[1,3,1,"","default_schema"],[1,4,1,"","ds"],[1,3,1,"","dump"],[1,4,1,"","dx"],[1,4,1,"","ep_mesh"],[1,4,1,"","f0"],[1,4,1,"","f0_ep"],[1,4,1,"","ffun"],[1,3,1,"","from_files"],[1,3,1,"","from_geometry"],[1,4,1,"","marker_functions"],[1,4,1,"","mechanics_mesh"],[1,4,1,"","mesh"],[1,4,1,"","microstructure"],[1,4,1,"","microstructure_ep"],[1,4,1,"","n0"],[1,4,1,"","n0_ep"],[1,4,1,"","num_refinements"],[1,4,1,"","outdir"],[1,4,1,"","s0"],[1,4,1,"","s0_ep"],[1,3,1,"","validate"]],"simcardems.geometry.LeftVentricularGeometry":[[1,3,1,"","default_markers"],[1,3,1,"","default_parameters"],[1,3,1,"","validate"]],"simcardems.geometry.SlabGeometry":[[1,3,1,"","default_markers"],[1,3,1,"","default_parameters"],[1,3,1,"","validate"]],"simcardems.land_model":[[1,1,1,"","LandModel"],[1,1,1,"","Scheme"]],"simcardems.land_model.LandModel":[[1,4,1,"","As"],[1,4,1,"","Aw"],[1,4,1,"","Ta"],[1,3,1,"","Wactive"],[1,4,1,"","Zetas"],[1,4,1,"","Zetaw"],[1,4,1,"","cs"],[1,4,1,"","cw"],[1,4,1,"","dLambda"],[1,4,1,"","dt"],[1,3,1,"","start_time"],[1,4,1,"","t"],[1,3,1,"","update_Zetas"],[1,3,1,"","update_Zetaw"],[1,3,1,"","update_prev"],[1,3,1,"","update_time"]],"simcardems.land_model.Scheme":[[1,2,1,"","analytic"],[1,2,1,"","bd"],[1,2,1,"","fd"]],"simcardems.mechanics_model":[[1,1,1,"","ContinuationBasedMechanicsProblem"],[1,1,1,"","MechanicsProblem"],[1,1,1,"","RigidMotionProblem"],[1,5,1,"","create_slab_problem"],[1,5,1,"","resolve_boundary_conditions"]],"simcardems.mechanics_model.ContinuationBasedMechanicsProblem":[[1,3,1,"","solve"],[1,3,1,"","solve_for_control"]],"simcardems.mechanics_model.MechanicsProblem":[[1,3,1,"","solve"],[1,4,1,"","u_subspace_index"],[1,3,1,"","update_lmbda_prev"]],"simcardems.mechanics_model.RigidMotionProblem":[[1,3,1,"","rigid_motion_term"],[1,4,1,"","u_subspace_index"]],"simcardems.newton_solver":[[1,1,1,"","MechanicsNewtonSolver"],[1,1,1,"","MechanicsNewtonSolver_ODE"]],"simcardems.newton_solver.MechanicsNewtonSolver":[[1,3,1,"","check_overloads_called"],[1,3,1,"","converged"],[1,3,1,"","default_solver_parameters"],[1,3,1,"","solve"],[1,3,1,"","solver_setup"]],"simcardems.newton_solver.MechanicsNewtonSolver_ODE":[[1,3,1,"","update_solution"]],"simcardems.postprocess":[[1,5,1,"","extract_biomarkers"],[1,5,1,"","extract_last_beat"],[1,5,1,"","extract_traces"],[1,5,1,"","find_decaytime"],[1,5,1,"","find_duration"],[1,5,1,"","find_ttp"],[1,5,1,"","get_biomarkers"],[1,5,1,"","json_serial"],[1,5,1,"","make_xdmffiles"],[1,5,1,"","numpyfy"],[1,5,1,"","plot_peaks"],[1,5,1,"","plot_population"],[1,5,1,"","plot_state_traces"],[1,5,1,"","save_popu_json"],[1,5,1,"","stats"]],"simcardems.save_load_functions":[[1,5,1,"","check_file_exists"],[1,5,1,"","decode"],[1,5,1,"","dict_to_h5"],[1,5,1,"","group_in_file"],[1,5,1,"","h5_to_dict"],[1,5,1,"","h5pyfile"],[1,5,1,"","load_initial_conditions_from_h5"],[1,5,1,"","load_state"],[1,5,1,"","mech_heart_to_bnd_cond"],[1,5,1,"","save_state"]],"simcardems.setup_models":[[1,1,1,"","EMState"],[1,1,1,"","Runner"],[1,1,1,"","TimeStepper"],[1,5,1,"","create_progressbar"],[1,5,1,"","setup_EM_model"],[1,5,1,"","setup_ep_solver"],[1,5,1,"","setup_mechanics_solver"]],"simcardems.setup_models.EMState":[[1,2,1,"","coupling"],[1,2,1,"","geometry"],[1,2,1,"","mech_heart"],[1,2,1,"","solver"],[1,2,1,"","t0"]],"simcardems.setup_models.Runner":[[1,3,1,"","create_time_stepper"],[1,4,1,"","dt_mechanics"],[1,3,1,"","from_models"],[1,4,1,"","outdir"],[1,3,1,"","save_state"],[1,3,1,"","solve"],[1,3,1,"","store"],[1,4,1,"","t"],[1,4,1,"","t0"]],"simcardems.setup_models.TimeStepper":[[1,4,1,"","T"],[1,4,1,"","dt"],[1,3,1,"","ms2ns"],[1,3,1,"","ns2ms"],[1,3,1,"","reset"],[1,4,1,"","total_steps"]],"simcardems.utils":[[1,1,1,"","MPIFilt"],[1,5,1,"","compute_norm"],[1,5,1,"","enum2str"],[1,5,1,"","float_to_constant"],[1,5,1,"","getLogger"],[1,5,1,"","local_project"],[1,5,1,"","print_mesh_info"],[1,5,1,"","remove_file"],[1,5,1,"","setup_assigner"],[1,5,1,"","sub_function"]],"simcardems.utils.MPIFilt":[[1,3,1,"","filter"]],simcardems:[[1,1,1,"","Config"],[1,1,1,"","DataCollector"],[1,1,1,"","DataLoader"],[1,1,1,"","EMCoupling"],[1,1,1,"","LandModel"],[1,1,1,"","MechanicsNewtonSolver"],[1,1,1,"","MechanicsNewtonSolver_ODE"],[1,1,1,"","MechanicsProblem"],[1,0,0,"-","ORdmm_Land"],[1,1,1,"","RigidMotionProblem"],[1,1,1,"","Runner"],[1,1,1,"","TimeStepper"],[1,0,0,"-","cli"],[1,0,0,"-","config"],[1,0,0,"-","datacollector"],[1,5,1,"","default_parameters"],[1,0,0,"-","em_model"],[1,0,0,"-","ep_model"],[1,0,0,"-","geometry"],[1,0,0,"-","land_model"],[1,0,0,"-","mechanics_model"],[1,0,0,"-","newton_solver"],[1,0,0,"-","postprocess"],[1,0,0,"-","save_load_functions"],[1,0,0,"-","setup_models"],[1,0,0,"-","utils"],[1,0,0,"-","version"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:property","5":"py:function"},terms:{"0":[1,2,3,4,13,14,15,16],"00":[2,3],"000000e":2,"001000e":2,"01":[1,2,3],"010000e":2,"015389":2,"02":[1,2,3,4,14],"03":[2,3],"04":[2,3],"05":[1,2,3,15,16],"06":2,"0648726":2,"07":[2,3],"070658":2,"08":[2,3],"09":2,"0c195ff":2,"1":[0,1,2,3,9,13,14,15,16,17],"10":[2,3,4,14],"100":[3,4],"1000":[1,2,4,5,15,16],"100000e":2,"102":3,"106":3,"11":[2,3],"110":3,"113":3,"118":3,"12":[2,3],"120":3,"123":3,"125":3,"13":[2,3],"133":3,"138":2,"14":[2,3,6],"140":1,"144":3,"146":3,"15":[2,3],"150":3,"16":[2,3],"169":3,"17":[2,3,6],"170":3,"172470c":2,"176":3,"18":3,"183":3,"19":[2,3],"1902":13,"193":3,"194":3,"1e":1,"2":[1,2,3,4,6,9,13,14,15],"20":[1,2,4,15,16],"200":[3,14],"2000":13,"2004":13,"2009":13,"2011":[3,9],"2014":6,"2017":3,"2022":[2,9],"21":3,"22":3,"226":3,"227464":2,"23":3,"230901":2,"238":3,"24":[2,3],"242":3,"249432":2,"25":[2,3],"250":3,"26":3,"260":3,"262":3,"266":3,"27":3,"276":3,"28":[2,3],"288":3,"29":3,"292":3,"2j":13,"3":[1,2,3,6,13,14],"30":4,"300":3,"301080":2,"31":3,"32":2,"323":3,"32e43ff":2,"33":3,"333":3,"34":[2,3],"340":3,"3445":13,"3475":13,"35":3,"350":3,"355":3,"357":3,"36":3,"361561":2,"367":13,"37":3,"375":3,"38":2,"386":3,"39":3,"3d":6,"4":[0,1,2,3,14],"40":2,"400":3,"41":[2,3],"42":2,"43":3,"430":3,"435":3,"438":3,"45":[2,3],"46":[2,3],"460":3,"470":3,"480":3,"49":3,"491":3,"493":3,"4_":13,"4_f":13,"4cc3e7a":2,"4d3a653":2,"5":[1,2,3,15,16],"50":3,"500":3,"5000":2,"501":13,"51":2,"516":3,"518":3,"52":3,"522":13,"523948":2,"53":3,"545":3,"55":[2,3],"552":3,"56":3,"562":3,"566156":2,"57":3,"592":3,"6":[2,3,6,16],"600":3,"603":3,"604":6,"609":3,"61":3,"615":6,"622805":2,"629":3,"63":3,"630ed09":2,"631049":2,"639097":2,"64":3,"648":3,"65":3,"653678":2,"661":3,"675":3,"68":3,"680":3,"688471":2,"68926df":2,"68ae7bb":2,"7":[2,3],"700":3,"708":3,"711879":2,"72":3,"74":3,"75":3,"750":3,"76":3,"78":3,"785172":2,"8":[2,3],"800":3,"805":3,"81":3,"82":3,"821":3,"83":3,"830650":2,"85":13,"850":3,"8501":[5,8],"86":3,"865697":2,"87":3,"870":3,"8_":13,"9":[2,3],"900":3,"92":3,"920":3,"93":3,"94":3,"97e958d":2,"983687":2,"99":3,"9d5c1b1":2,"\u00e1":3,"\u00f3":3,"abstract":1,"boolean":4,"c\u00e9cile":9,"case":[3,6,13,14,15,16],"class":[1,14],"default":[1,4,8,17],"do":[0,1,3,5,11,14],"enum":1,"final":[15,16],"float":[1,4],"function":[1,13,14],"import":[2,6,13,14,15,16],"int":1,"new":[4,5,14],"public":0,"return":[1,2,14],"static":1,"super":14,"true":[1,2,14,15,16],"try":8,"while":[6,13],A:[3,6,13,17],And:14,As:[0,1,3],Being:0,By:[8,17],For:[4,5,6,13,17],If:[0,1,4,5,8,12,14],In:[0,6,8,14,15,16],It:2,On:12,One:[4,13],The:[0,1,4,6,8,9,11,13,14,15,16,17],There:6,These:[8,11,17],To:[0,5,6,8,13],_1:13,_2:13,_:13,__file__:[14,15,16],__init__:14,__main__:14,__name__:14,_a:13,_p:13,_post_mechanics_solv:14,_print_messag:14,_t_releas:14,_time_stepp:14,a_:[3,13],a_f:13,ab:[2,13],abc:1,abl:13,about:[6,8,11,13,15],absolut:[2,14,15,16],abus:0,acamk:3,acap:3,accept:0,account:0,act:0,action:[0,3],activ:[1,6,12,14,15,16,17],active_model:1,activemodel:1,ad:12,adapt:0,add:[12,14,16],add_trac:2,addit:[0,1],address:0,advanc:0,af:3,afca:3,afcaf:3,aff:3,aforement:10,after:[5,14,17],ag:0,ageo:3,ah:3,ahead:5,ahf:3,ai:3,aif:3,aka:5,alexand:13,alia:1,align:[0,13,14],all:[0,2,4,8,10,11,17],all_hash:2,allo_:3,allow:0,along:13,alreadi:1,also:[0,4,6,8,10,12,14,15,16,17],altern:[11,12],although:4,alwai:13,amd64:11,among:1,amp:3,an:[0,1,8,11,14],anaconda:12,analyt:[1,2,4,15,16],anca:3,andr:3,andrew:6,angular:13,ani:[0,1,6,9,13],anymor:5,anyth:4,ap:3,api:[9,15],app:5,appear:0,append:2,appl:12,appli:[0,14],appoint:0,approach:13,appropri:[0,1,12],apt:12,ar:[0,2,5,6,9,10,11,12,13,15,16,17],architectur:11,arg0:1,arg1:1,arg2:1,arg3:1,arg4:1,arg:[1,4],argument:[4,14],arm64:11,arrhythmia:13,as_dict:[1,15,16],ask:0,ass:3,assert:14,assp:3,assum:4,attack:0,attent:0,autogener:1,automat:[3,8],avail:[0,10],averag:[1,16,17],aw:[1,3],axr:3,axrf:3,b:[1,3,13],b_:[3,13],b_f:13,back:6,background:5,ban:0,bar:4,base:[1,3,4,8,9,10,13,16],basegeometri:1,bcai:3,bcajsr:3,bcamk:3,bcass:3,bd:[1,4],becaus:[2,5],becom:4,been:5,befor:[0,12],begin:[3,13,14],behavior:0,benchmark:[1,4,9],benchmark_fold:2,best:0,better:1,between:[14,17],big:4,biologi:[3,13],biomark:2,biomechan:6,biomed:6,biophys:13,bnd_cond:4,bnd_rigid:[1,15,16],bodi:0,bool:1,both:[0,6,10,11,12,13],boundari:[1,4,14,16],boundarycondit:1,br:2,branch:0,browser:8,bslmax:3,bsrmax:3,bt:3,btp:3,built:[11,12],button:8,c7df02b:2,c:[2,3,12,13],c_m:1,ca:[1,14,15,16,17],cai:3,cajsr:3,calcium:[14,15,16,17],calib:3,call:[4,6,8,13,14,15,16,17],camka:3,camkb:3,camko:3,camkt:3,can:[0,2,4,5,6,8,9,10,11,12,13,14,15,16,17],cansr:3,cao:3,cardiac:[1,3,6,10,13],cardiac_geometri:[1,15,16],cardiaccellmodel:1,cardiolog:3,cardiomyocit:13,cardiomyocyt:3,cass:3,cat_:3,catrpn:3,catrpn_:3,catti:9,cauchi:13,cbcbeat:[1,7,9,10],cd:[3,12],cdot:13,cecil:9,cell:[1,6],cell_init:1,cell_init_fil:[1,4,15,16],cell_param:1,cellmodel:1,celltyp:3,cellular:[6,9,13],center:[2,14,15,16,17],cf7cc9d:2,chang:[0,2,4,8,14],channel:6,character:13,check_file_exist:1,check_overloads_cal:1,checkbox:0,checkout:[0,2],chi:1,chip:12,choic:[11,13],choos:4,ci:12,circumst:0,clarifi:0,classmethod:1,click:[0,8],clone:12,cluster:[4,12],cmdnmax:3,coarser:6,code:[2,4,12],col:2,collabor:0,collect:2,column:2,com:[2,12],come:[12,13],command:[8,9,11,12,17],comment:0,commit:[0,2,12],common:[6,12,13],commun:0,complaint:0,complic:16,compon:6,comput:[3,4,5,6,8],computationalphysiolog:[2,5,8,11,12,15,16],compute_norm:1,concentr:[14,15,16,17],conda:12,condit:[1,4,14,16],conf:[1,14],confidenti:0,config:[14,15,16],configur:[13,15,16],conflict:0,consequ:6,conserv:13,consid:0,constant:[1,14],constitut:13,construct:0,consult:8,contact:0,contain:[1,4,6,8,10,11,15,16,17],content:4,continu:[1,2,4],continuationbasedmechanicsproblem:1,contract:[3,6,13],contribut:[9,10,12,13],control:[1,14],converg:[1,6],convers:0,convert:[1,14],copyright:9,core:9,correct:0,correctli:[2,14],correspond:17,could:0,coupl:[1,9,10,13,14],coupling_to_ep:1,coupling_to_mechan:1,coven:0,coverag:0,cpp:1,cpu:12,creat:[0,4,8,11,12,14,15,16],create_boxmesh:1,create_progressbar:1,create_slab_facet_funct:1,create_slab_microstructur:1,create_slab_problem:1,create_time_stepp:1,creation:17,cristob:3,critic:0,cs:[1,3],csqnmax:3,current:[1,15,16],custom:[4,15,16],cw:[1,3],cycl:4,d:[1,2,3,6],da:3,daemon:5,dap:3,data:[1,2,9],datafram:2,datagroup:1,dataload:1,date:[2,12],daversin:9,dbbec69:2,dcai:3,dcajsr:3,dcamkt:3,dcansr:3,dcass:3,dcatrpn:3,dcd:3,dd:3,debug:4,declar:13,decod:1,decompos:13,dedic:[10,12],deem:[0,1],def:[2,14],default_conduct:1,default_initial_condit:1,default_mark:1,default_microstructur:1,default_paramet:1,default_schema:1,default_solver_paramet:1,defaultdict:2,defin:[0,6,13],define_conductivity_tensor:1,deform:13,delet:8,delta:[1,3],delta_:3,demo:[4,16],densiti:13,depend:[6,9,10,14],deriv:13,derogatori:0,describ:6,det:13,detail:[0,2,13],determin:[0,1],dev:12,develop:[3,8,9,10],df9ad53:2,df:[2,3],dfca:3,dfcaf:3,dfcafp:3,dff:3,dffp:3,dh:3,dhf:3,dhl:3,dhlp:3,dhsp:3,di:3,dict:1,dict_to_h5:1,dictionari:1,dif:3,differ:[0,2,8,15,16],difp:3,direct:[13,17],directli:[0,12],directori:[4,8,15,16],dirichlet:[1,4,14],disabl:0,discret:6,discretis:6,discuss:[0,6],diseas:4,disease_st:[1,4,15,16],disp:3,displac:[13,14,15,16],displai:[2,4],dit:5,dj:3,djca:3,djp:3,djrelnp:3,djrelp:3,dki:3,dkss:3,dlambda:[1,3],dm:3,dml:3,dnai:3,dnass:3,dnca:3,docker:[9,10,12],dockerfil:11,document:[0,3],documentaion:9,doe:[1,2],doesn:5,dolfin:[1,14],domain:[1,16,17],don:[0,5],done:[5,8],drop:[14,15,16],drug:4,drug_factors_fil:[1,4,15,16],ds:1,dss:3,dt:[1,3,4,15,16],dt_mechan:1,dti_:3,dtmb:3,dump:1,durat:3,dure:14,dv:[1,3],dx:[1,3],dxk_:3,dxr:3,dxrf:3,dxs_:3,dxw:3,dzeta:3,dzetaw:3,e1002061:3,e1:3,e:[0,2,3,4,6,8,12,13,14],e_:3,each:[0,4,6,8,15,16,17],edc973c:2,edit:[0,12],either:[0,1,8],ek:3,electr:6,electromechan:13,electron:0,electrophysiolog:[6,9,10,14],element:[10,13],email:0,emcoupl:[1,3],empathi:0,emploi:13,empti:1,emstat:1,en:7,ena:3,enabl:0,encompass:6,end:[3,4,13,14],energi:[1,13],enforc:13,engin:[6,13],enough:4,ensur:6,enum2str:1,enumcl:1,enumer:[1,2],environ:[0,11],ep:[1,3,4,8],ep_mesh:1,ep_ode_schem:[1,15,16],ep_precondition:[1,15,16],ep_solv:1,ep_theta:[1,15,16],ep_to_coupl:1,epi:3,equat:[2,6,13],equilibrium:13,esac_:3,eta:[1,3],etal:3,ethnic:0,evalu:2,event:0,everyon:0,exampl:[0,4,5],excit:13,exec:5,execut:[4,8,11,17],exist:[1,2,4],exit:4,expect:0,experi:0,experiment:3,explicit:0,express:0,exterior:1,extern:13,extra:[8,14],extract:17,extract_biomark:1,extract_last_beat:1,extract_trac:1,extract_valu:1,f0:1,f0_ep:1,f63afc9:2,f:[1,2,3,13,14],f_:3,face:0,facet:1,factor:4,fail:14,failur:1,fair:0,faith:0,fallingfactori:3,fals:[1,2,14,15,16],fca:3,fcaf:3,fcafp:3,fcap:3,fcass:3,fd:[1,3,4],feedback:6,fenic:[4,9,10],fenics_plotli:8,fenicsproject:12,ff:3,ffp:3,ffun:1,ffun_path:1,fiber:13,fiber_spac:1,ficalp:3,field:[1,14],fig:2,figur:[14,15,16,17],file:[1,4,8,14,15,16],file_exist:1,filemod:1,filenam:1,fill:0,filter:1,finalp:3,finap:3,find:0,find_decaytim:1,find_dur:1,find_ttp:1,finish:0,finit:[10,13],finsberg:9,first:[0,5,8,13,14],fitop:3,fix:[14,16],fix_right_plan:[1,15,16],fjrelp:3,fjupp:3,float_to_const:1,flow:[0,14],fname:1,focus:0,folder:[4,8,11,17],follow:[0,4,5,8,11,12,15,16],forc:14,forg:12,forget:0,fork:0,formatt:12,formul:3,forward:8,foster:0,found:9,fp:3,frac:[3,13],framework:[10,13],free:0,frequenc:4,from:[0,1,2,3,6,8,11,12,13,14,15,16,17],from_fil:1,from_geometri:1,from_model:1,fs:[3,13],fss:3,further:[0,6],futhermor:6,futur:16,g:[3,4],gamma:3,gamma_0:14,gamma_1:14,gammasu:3,gammaw:3,gammawu:3,gender:0,gener:[2,3,16,17],genericmatrix:1,genericvector:1,geo:1,geometri:[4,6,9,14,15,17],geometry_path:[1,4,14,15,16],geometry_schema_path:[1,14,15,16],geq:14,gerhard:13,get:[0,1,8,11,12,14,17],get_biomark:1,get_ylim:2,getlogg:1,ghcr:[5,8,11],git:[0,2,12],git_commit_hash:2,git_hash:2,github:[0,2,9,11,12,15,16],given:13,gk:3,gk_:3,gkb:3,gkr:3,gna:3,gnal:3,gncx:3,gnu:9,go:[2,8],good:[0,4,8,11],gotran:[1,3],gpca:3,gracefulli:0,gradient:13,graph_object:2,graphic:9,greater:14,green:13,grl1:[1,15,16],group:1,group_in_fil:1,gsac_:3,gto:3,guccion:4,guess:1,gui:4,h5:[4,14,15,16,17],h5_filenam:1,h5_to_dict:1,h5group:1,h5name:1,h5path:1,h5pyfil:1,h:[2,3],h_:3,ha:[4,5,6,17],half:14,hand:1,handle_cell_init:1,handle_cell_param:1,happen:6,hara:3,harald:6,harass:0,hard:2,harm:0,hash:[2,8],have:[0,5,6,12,14],hca:3,healthi:[1,4,15,16],heart:[1,13],height:2,help:[0,4,5],henrik:9,henriknf:[0,9],herck:9,here:[0,2,8,13,14,15,16],hf:[1,3],hi:4,hide:4,hide_progress_bar:4,high:[4,6,12],hl:3,hlp:3,hlss:3,hlssp:3,hna:3,ho09:13,hol00:13,holohan:3,holzapfel:13,holzapfelogden:4,home:[8,12],hood:13,hook:12,hovertempl:2,how:[2,4,5,8,11],howev:2,hp:3,hpc:[4,12],hs:3,hsp:3,hss:3,hssp:3,html:[7,16],http:[0,2,7,8,12,15,16],human:3,hyperelast:13,i:[1,2,3,6,8,13,14],iF:3,iS:3,i_1:13,i_:13,ic:4,icab:3,icak:3,icana:3,id:8,idea:[8,14],ident:0,ifp:3,ii:13,ik1:3,ik:3,ik_:3,ikb:3,ikr:3,ils:9,ilsbeth:9,imag:[5,12],imageri:0,impact:2,implement:[2,4,6,14],import_tim:2,improv:6,ina:3,inab:3,inaca_:3,inak:3,inal:3,inappropri:0,inc:13,incid:0,includ:[0,10],inclus:0,incompress:13,index:1,indic:4,individu:[0,8],inf:3,info:[4,13,15,16],inform:[0,6,7],infp:3,init_condit:1,initi:[1,4],input:8,insid:8,instal:[4,5,8],instanc:0,instant:[14,15,16],instead:16,instruct:[11,12],insult:0,integ:4,integr:6,intel:12,interact:0,intercellular:[14,15,16],interest:0,interfac:9,intern:14,interpol:6,intracellular:17,inv_lmbda:1,invari:13,investig:0,involv:6,io:[5,7,8,11,12,15,16],ip:3,ipca:3,is_fil:2,isac:3,isac_:3,isclos:2,isinst:14,isol:11,isp:3,iss:3,issu:0,istim:3,item:2,iter:[4,14],iterdir:2,ito:3,its:[0,4],itself:13,j:[3,13],j_:3,jca:3,jdiff:3,jdiffk:3,jdiffna:3,jin:3,jleak:3,jnakk:3,jnakna:3,jncxca_:3,jncxna_:3,joakim:6,john:13,join:2,joinpath:[14,15,16],jonathan:3,journal:3,jp:3,jrel:3,jrel_:3,jrelnp:3,jrelp:3,json:[2,4,14,15,16],json_seri:1,jss:3,jtr:3,jup:3,jupnp:3,jupp:3,k1m:3,k1p:3,k2m:3,k2n:3,k2p:3,k3m:3,k3p:3,k3p_:3,k3pp:3,k3pp_:3,k4m:3,k4p:3,k4p_:3,k4pp:3,k4pp_:3,k:3,k_:3,kasymm:3,kb:3,kcaoff:3,kcaon:3,keep:2,kei:2,kentish:3,keyerror:1,khp:3,ki:3,kirchhoff:13,kki:3,kko:3,km2n:3,kmbsl:3,kmbsr:3,kmcaact:3,kmcam:3,kmcamk:3,kmcmdn:3,kmcsqn:3,kmgatp:3,kmn:3,kmtrpn:3,kna_:3,knai:3,knai_:3,knao:3,knao_:3,knap:3,know:13,known:0,ko:3,ksca:3,kss:3,ksu:3,ktrpn:3,ku:3,kuw:3,kw:3,kwarg:1,kwu:3,kxkur:3,l:3,l_x:14,la:1,laboratori:9,lagrang:13,lambda:[3,13,14,15,16,17],lambda_:3,land:3,landmodel:1,languag:0,later:9,latest:[7,11,12],law:13,leadership:0,left:[3,9,13],leftventriculargeometri:1,legaci:[10,12],len:2,length:[4,14],leq:3,level:[0,6,9],lgpl:9,librari:[2,15,16],lightweight:8,like:14,line:[8,9],linear:13,linear_mechanics_solv:[1,15,16],linear_solv:1,linearis:6,link:[0,2],linter:12,list:[1,2,8],lmbda:[1,3],lmbda_prev:1,load:[2,4,13,16],load_geometri:1,load_initial_conditions_from_h5:1,load_json:1,load_stat:[1,4,15,16],loader:1,local_project:1,localhost:8,locat:8,log:[1,3,14],loglevel:[1,4,15,16],look:[4,8],loss:6,lot:8,lower:0,lph:3,lv_ellipsoid:16,lx:[1,14],ly:1,lz:1,m:[3,4,5,8,12,17],machin:5,made:0,mai:[0,1],mail:0,main:[1,6,14],maintain:0,make:[0,2,5,13,14,17],make_subplot:2,make_xdmffil:[1,15,16],manual:[8,12],mark:0,marker:1,marker_funct:1,marker_path:1,martyn:13,materi:[1,4,13],material_model:1,mathbb:13,mathbf:[13,14],mathcal:3,mathemat:13,mathrm:13,matplotlib:2,max:[1,2,3],max_column:2,mcculloch:6,mean:2,measur:[1,3],mech_heart:[1,14],mech_heart_to_bnd_cond:1,mech_mesh:1,mechan:[1,4,8,10,14],mechanics_mesh:1,mechanics_ode_schem:[1,15,16],mechanics_to_coupl:1,mechanics_use_continu:[1,15,16],mechanics_use_custom_newton_solv:[1,15,16],mechanicsnewtonsolv:1,mechanicsnewtonsolver_od:1,mechanicsproblem:1,mechano:6,media:0,member:0,membran:[1,6,14,15,16,17],merg:0,mesh:[1,4,8],mesh_path:1,meshfunct:1,messag:[4,14],method:[0,6,13,14],mgadp:3,mgatp:3,microstructur:1,microstructure_ep:1,microstructure_path:1,middl:14,might:[2,4],millisecond:[1,14],min087:3,min12:3,min:[1,2],ml:3,mlss:3,mode:[3,5],model:[1,4,8,9,14],modifi:1,modul:[7,15,16],molecular:[3,13],momentum:13,monodomain:[6,7],monodomainsolv:7,more:[2,7,8,11,13,15,16],most:[12,14,15],motion:[1,13],mpi:4,mpifilt:1,mpirun:4,ms2n:1,ms:[2,4],mss:3,much:[4,6],multipli:13,mump:[1,15,16],myocardium:13,n0:1,n0_ep:1,n:[2,4,12],nabla:13,nai:3,name:[1,2,3,4,5,11,17],namedtupl:1,nanosecond:[1,14],nao:3,nash:13,nass:3,nation:0,natur:13,nca:3,necessari:[0,14,15,16],need:[5,6,8,12],neumann:4,newton:[1,4],newtonsolv:1,next:[14,15,16],nice:[0,4],nicola:3,nieder:3,nl:1,node:6,non:13,none:[1,4,14,15,16],nonetyp:1,nonlinear:13,nonlinearproblem:1,note:[0,4,5,12,13,14,16],noth:1,novel:3,now:[5,14],np04:13,np:[2,4],ns2m:[1,14],ns:3,ntm:3,ntrpn:3,num_model:1,num_refin:[1,4,15,16],num_stat:1,number:[1,4],numer:6,numpi:2,numpyfi:1,o:[3,4,14],obj:1,object:1,oblig:0,obtain:[4,6],od:[1,4,6,14],off:14,offens:0,offer:4,offici:0,offlin:0,ogden:13,oharaviragvarror11:3,omega:13,onc:[0,4,5,12,14],one:[0,4,6,13,14],onli:[4,12,14,16],onlin:0,open:[0,2,8,10,15,16,17],operatornam:3,opposit:14,option:[1,2,4,8,9,10,12],order:[2,5,6,8,14],ordereddict:1,ordmm_land_em_coupl:1,org:0,orient:0,origin:[13,17],os:[1,12],osn:6,other:[0,6,14],otherwis:[0,1,3,4],out:2,outdir:[1,4,14,15,16],output:[4,8,14,15,16],over:[2,16,17],overview:0,owner:0,p:[3,5,8,13],p_:3,pace:[1,4],packag:[5,8,10,12],page:[0,8],panda:2,panfilov:13,paper:13,parallel:4,param:1,paramet:[1,8,14],parameter_path:1,paraview:[15,16],parent:[2,14,15,16],park:3,part:[9,14],partial:13,particip:0,pass:[0,6],path:[1,2,4,14,15,16],path_to_result:17,pathlib:[1,2,14,15,16],pathlik:1,pca:3,pcab:3,pcak:3,pcakp:3,pcana:3,pcanap:3,pcap:3,pcl:[1,4,15,16],pd:2,perform:[4,12],perman:0,permiss:0,person:0,phenomena:6,phi:3,phicak:3,phicana:3,philosoph:13,physic:[0,13],piola:13,pip:[8,9,10],pkna:3,place:1,plan:16,pleas:[0,5,8,15],plo:3,plot:[8,14,15,16],plot_peak:1,plot_popul:1,plot_state_trac:[1,14,15,16],plotli:2,plt:2,pnab:3,pnak:3,png:[15,16],point:[15,16],polici:0,polit:0,popu_factors_fil:[1,4,15,16],popul:4,population_fold:1,port:8,posit:[0,14],posixpath:[15,16],possibl:12,post:0,postprocess:[4,14,15,16,17],potenti:[1,3,6,14,15,16,17],ppa:12,pprint:[15,16],pr:0,pre:[11,12,14],pre_stretch:[1,14,15,16],prebuilt:12,precis:6,precondition:1,prefer:11,present:17,previou:[4,12],prima:3,princip:13,print:[2,4,14,15,16],print_mesh_info:1,privat:0,problem:[1,4],process:6,processor:4,profession:0,progress:[4,13],project:[0,9,11],properti:[1,4,12],propos:0,provid:[1,4,11,12],psi:13,psi_a:13,psi_p:13,publish:0,pull:11,puls:[1,9,10,13,14],pure:10,purpos:0,pwd:[5,12],pypi:10,pyplot:2,python3:[4,5,8,12,17],python:[4,10,12,15],qca:3,qna:3,quai:12,quantiti:13,question:0,quick:17,quickstart:16,quit:8,r:[1,2,3],race:0,rad:3,radio:8,rai:13,rais:1,raise_on_fals:1,rang:2,rapid:14,re:0,read:[0,11],reader:13,readi:0,readthedoc:7,reason:0,rebuild:5,recommend:[4,10,12],recomput:6,record:1,recov:3,redistribut:1,reduct:[1,14,17],reentrant:13,ref:3,refer:12,refin:[1,4],refine_mesh:1,regard:0,regardless:0,regist:1,register_ep_model:1,register_mech_model:1,registri:11,reject:0,rel:3,rel_max_displacement_perc_z:2,relat:13,releas:[9,15,16],release_test_result:14,releaserunn:14,relev:[6,13],reli:4,religion:0,relp:3,remedio:3,remov:0,remove_fil:1,render:0,repercuss:0,repo:[11,12],repolaris:1,report:0,repositori:[0,12],repres:[0,1,6,13],represent:0,requir:[6,8,10,17],rerun:2,research:9,reset:1,reset_st:1,reset_tim:1,resolut:6,resolv:0,resolve_boundary_condit:1,resourc:[0,5],respect:[0,13],result:[0,1,2,4,14,15,16,17],results_:8,results_fil:[1,2],results_lv_ellipsoid:16,results_rigid:4,results_simple_demo:[15,16],retriev:1,return_interv:1,review:0,right:[0,1,3,13],rigid:[1,4],rigid_motion_term:1,rigidmotionproblem:1,rk_:3,rkr:3,rm:[5,8],robin:4,root:11,row:2,royal:13,rs:3,rudi:3,run:[0,2,4,9,11,12,16],runner:[1,14,15,16],rw:3,s0:1,s0_ep:1,s:[0,1,3,4,6,13],same:[1,2,4,16,17],sander:3,satisfi:13,save:[4,15,16,17],save_freq:[1,4,15,16],save_popu_json:1,save_st:1,scalabl:4,scale:4,scale_:3,scatter:2,schema:[4,16],schema_path:1,scheme:[1,4,6,15,16],scienc:13,seamlessli:4,section:6,see:[0,1,2,4,7,14,15,16],select:8,self:[1,14],separ:[0,6,8],set:[0,1,4,12,14,15,16],set_log_level:14,set_materi:[1,4,15,16],setup:[1,8],setup_assign:1,setup_em_model:1,setup_ep_model:1,setup_ep_solv:1,setup_mechanics_solv:1,setup_splitting_solver_paramet:1,sever:6,sexual:0,shall:4,share:12,shared_xax:2,sheet:13,should:[1,5,12],show:[0,2,4,15,17],show_progress_bar:[1,4,15,16],showlegend:2,shown:[15,16],side:[1,14],signific:6,silicon:12,simcardem:[2,4,5,8,9,10,11,14,15,16,17],simcardems_vers:2,simcardio:9,simpl:[2,9,16],simul:[2,3,4,6,14,15,16,17],simula:0,size:[0,1],slab:[2,4,13,14,15,16,17],slabgeometri:[1,14],slap:14,small:4,smith:3,so:[0,3,5,6,8,14],social:0,societi:13,softwar:[10,12],solid:13,solut:[2,6],solv:[0,1,4,13,14,15,16],solve_for_control:1,solver:[1,4,6,10,13,14],solver_setup:1,some:[0,6,8,14],someth:8,son:13,sor:[1,15,16],sort_valu:2,sourc:[0,1,10,12],space:0,spatial:6,speak:13,special:6,specif:[0,2,13,15,16],specifi:[1,4,15,16],split:13,splittingsolv:1,spring:[1,4,15,16],sqrt:[3,13],ss:3,st_progress:1,stabil:6,stabl:12,stamp:1,start:[0,15,16],start_tim:1,stat:1,state:[1,4,6,14,15,16,17],state_prev:1,state_trac:[15,16],step:[1,4,5,6,15,16,17],steven:3,storag:8,store:[1,8,14,17],str:1,strain:13,streamlit:8,strength:6,stress:[1,13,17],stretch:[6,13,14,15,16],string:1,strongli:6,structur:13,studi:13,sub_funct:1,subclass:14,subdomain_data:1,subfold:8,submit:0,subplot:2,subplot_titl:2,subsystem:12,sudo:12,suffix:[1,4],suggest:0,sum:13,sundn:6,support:[11,16],sure:[0,5,14],surfac:1,swap:2,swo:6,system:[1,12,14],szl:3,t0:1,t:[0,1,2,3,4,5,11,13,14,15,16],t_a:[13,14,15,16,17],t_releas:14,ta:[1,3],tab:[0,8],take:[0,4,5,8,14],taken:[2,13],tau_:3,taylor:13,td:3,team:0,templat:0,temporari:0,temporarili:0,tension:[3,6,13,14,15,16,17],tensor:13,test:[0,9,12],text:[2,3,4,14],tf:3,tfca:3,tfcaf:3,tfcafp:3,tff:3,tffp:3,th:3,than:0,thei:0,them:0,therefor:6,theta:1,thf:3,thi:[0,1,3,4,5,6,8,10,11,13,14,15,16,17],think:6,thl:3,thlp:3,thoma:3,thorvaldsen:6,threaten:0,three:6,threshold:1,through:[0,6,11,14],thsp:3,ti:[3,12],tif:3,tifp:3,time:[1,2,3,4,6,8,14,15,16],time_stamp:1,time_stepp:1,time_to_max_displacement_z:2,timestamp:2,timestepp:1,tisp:3,tissu:[6,9,17],tj:3,tjca:3,tjp:3,tm:3,tmb:3,tml:3,tmp:3,to_datetim:2,tol:1,tom:6,tot_:3,total:13,total_dof:1,total_step:1,toward:0,tp:3,tr:13,trace:[2,14,15,16],track:2,traction:[1,4,15,16],transact:13,transmembran:1,tref:3,trigger:[14,15,16],troll:0,troublesom:4,trpn:3,trpn_:3,trpnmax:3,ttot:3,tupl:1,turn:14,tutori:0,two:6,txk_:3,txr:3,txrf:3,txs_:3,type:[1,4,14],u:[1,13,14,15,16],u_subspace_index:1,ubuntu:12,ui:0,unaccept:0,under:[8,9],underli:14,understand:0,undiseas:3,union:1,unit:13,unlink:1,unwelcom:0,up:[0,1,5,8,12,14],updat:[0,5,12],update_cb:1,update_layout:2,update_lmbda_prev:1,update_prev:1,update_solut:1,update_tim:1,update_yax:2,update_zeta:1,update_zetaw:1,url:8,us:[0,1,3,4,7,8,9,10,11,12,13,15,16,17],usabl:12,usag:[4,15],use_custom_newton_solv:1,use_n:1,user:[9,12],v:[1,2,3,5,12,13,14,15,16,17],valid:[1,3],valu:[1,2,3,4,6,14,17],van:9,vari:14,variabl:[1,14,15,16,17],variat:1,varr:3,vcell:3,vector:13,ventricular:[3,9],veri:4,verifi:2,version:[0,2,4,9,10,12],vffrt:3,vfrt:3,via:[0,6],view:[15,16],viewpoint:0,vir:3,virtual:5,visual:[8,9,15,16],vjsr:3,vmyo:3,vnsr:3,voltag:[15,16],volum:1,vs:1,vs_functions_to_dict:1,vss:3,w:[1,12,13],wa:1,wactiv:1,wai:[0,6,11,14],wall:6,want:[1,4,5,8,10,11,12,15,16],warn:[4,14],wca:3,we:[0,2,4,6,7,8,10,11,12,13,14,15,16],welcom:0,well:0,what:[0,13],when:[0,4,5,14],whenev:2,where:[1,2,8,13,14],which:[0,4,6,8,10,11,12,13,14,15,16],whiteout:6,who:[0,10],wiki:0,wilei:13,wish:0,within:[0,11],without:[0,4,13],wna:3,wnaca:3,word:[6,14],work:3,workflow:0,would:14,wrap:1,x:[1,2,13,14],x_:3,x_prev:1,xdmf3reader:17,xdmf3readert:17,xdmf:[15,16],xk1ss:3,xk_:3,xkb:3,xmdf:17,xr:3,xrf:3,xrss:3,xs1ss:3,xs2ss:3,xs:[1,3],xs_:3,xu:3,xw:[1,3],xw_:3,y:[1,2,13],y_max:2,y_mean:2,y_min:2,yoram:3,you:[0,1,2,4,5,8,11,12,15,16,17],your:[0,5,8,9],yourself:11,yrang:2,zca:3,zero:14,zeta:[1,3],zetas_prev:1,zetaw:[1,3],zetaw_prev:1,zip:2,zk:3,zna:3},titles:["Contributing","API documentaion","Benchmark","Cellular model","Command line interface","Run using Docker","Electro-Mechanical coupling","Tissue level electrophysiology model","Graphical user interface","Simula Cardiac Electro-Mechanics Solver","Installation","Install with Docker","Install with <code class=\"docutils literal notranslate\"><span class=\"pre\">pip</span></code>","Tissue level mechanics model","Release test","Simple demo","Left ventricular geometry","Data visualization"],titleterms:{"new":0,activ:13,again:5,api:1,attribut:0,author:9,benchmark:2,between:6,build:11,cardiac:9,cellular:3,cli:1,code:[0,9],command:[4,5],commun:9,conduct:0,config:1,contain:5,content:[1,9],contribut:0,contributor:0,coupl:6,creat:[5,17],data:17,datacollector:1,delet:5,demo:[9,15],develop:12,differ:6,discret:13,docker:[5,8,11],documentaion:1,electro:[6,9],electrophysiolog:7,em_model:1,enforc:0,ep:6,ep_model:1,execut:5,express:3,fenic:12,file:17,geometri:[1,16],graphic:8,gui:8,guid:0,imag:11,instal:[9,10,11,12],interfac:[4,8],land_model:1,left:16,level:[7,13],licens:9,line:[4,5],linux:12,mac:12,mathemat:9,mechan:[6,9,13],mechanics_model:1,mesh:6,model:[3,6,7,13],modul:1,newton_solv:1,node:17,ordmm_land:1,ordmm_land_em_coupl:3,our:0,outsid:8,own:11,paramet:3,paraview:17,passiv:13,pip:12,pledg:0,plot:17,postprocess:[1,8,9],process:0,programm:9,pull:0,refer:[3,6,9,13],releas:14,request:0,respons:0,result:8,run:[5,8],save_load_funct:1,scope:0,script:5,setup_model:1,simcardem:[1,12],simpl:15,simul:8,simula:9,singl:17,softwar:13,solv:6,solver:9,sourc:9,standard:0,start:[5,8],state:3,stop:5,test:14,theori:9,tissu:[7,13],trace:17,transfer:6,us:5,user:8,util:1,variabl:6,ventricular:16,verif:9,version:1,visual:17,window:12,xdmf:17,your:11}})