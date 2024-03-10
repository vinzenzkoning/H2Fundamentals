# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:07:03 2023

@author: Konin045
"""



import plotly.graph_objects as go


#### colors #####
#006400 rgb(0,100,0) darkgreen   alpha = 0.7
#f5deb3 rgb(245,222,179) wheat alpha = 0.3
#696969 rgb(105,105,105) dimgray alpha = 0.7
#6b8e23 rgb(107,142,35) olivedrab alpha = 0.5
#808000 rgb(128,128,0) olive alpha = 0.7


# etaH2P=0.7
# r = 2.
# f = 0.15
# Edir =1-f
# Ecurt = 0.15
# Hexp = 0.65
# Ee= r - Edir - Ecurt 
# EH2P = 0.2 #adjust
# Eren = Edir + EH2P
# H2P_losses = etaH2P*(1-etaH2P)
# HP = etaH2P*etaH2P
# H = HP + Hexp



# =============================================================================
# 
# "r", 
# "f", 
# "curtailed", 
# "E_electrolyzer", 
# "losses electrolyzer", 
# "H", 
# "losses H2P", 
# "E_H2P", 
# "E_dir", 
# "E_ren", 
# "E_f", 
# "H_exp", 
# "D=1"
# =============================================================================


Eflows = {
  "r":0, 
  "f":1, 
  "curtailed":2, 
  "E_electrolyzer":3, 
  "losses":4, 
  #"H":5, 
  "HP":5,#6, 
  "E_H2P":6,#7, 
  "E_dir":7,#8, 
  "E_ren":8,#9, 
  "E_f":9,#10, 
  "H_exp":10,#11, 
  "D=1":11#12  
}

#=============================================================================
Eflows = {
  "r":0, 
  "f":1, 
  "curtailed":2, 
  "E_electrolyzer":3, 
  "losses":4, 
  "H":5, 
  "HP":6, 
  "E_H2P":7, 
  "E_dir":8, 
  "E_ren":9, 
  "E_f":10, 
  "H_exp":11, 
  "D=1":12  
}
#=============================================================================

#full power system decarbonization
etaH2P=0.5
etaP2H=0.7
r = 1.43
f = 0.0
Edir =1-f
Ecurt = 0.09
Ee= r - Edir - Ecurt 
P2H_losses = (1-etaP2H)*Ee
H = etaP2H*Ee
Hexp = 0.0
HP=H
H2P_losses = (1-etaH2P)*HP
EH2P = etaH2P * HP #adjust
Eren = Edir + EH2P

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 70,
      thickness = 10,
      line = dict(color = "gray", width = 0.0),
      #label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "H", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      #label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      color =  ["#C1CDCD", "rgb(107,142,35)", "rgb(245,222,179)", "rgb(107,142,35)", "rgb(107,142,35)", "#C1CDCD", "#C1CDCD", "rgba(128,128,0,0.7)", "rgba(0,100,0,0.7)", "#C1CDCD", "#C1CDCD", "#C1CDCD","#C1CDCD"]
    ),
    link = dict(    
      source = [Eflows["r"], Eflows["r"], Eflows["r"], Eflows["f"], 
                Eflows["E_dir"], Eflows["E_ren"], Eflows["E_f"], Eflows["E_electrolyzer"], 
                Eflows["E_electrolyzer"], Eflows["H"], Eflows["H"], Eflows["HP"], 
                Eflows["HP"], Eflows["E_H2P"]], # indices correspond to labels,
      #source = [0, 0, 0, 1, 8, 9, 10, 3, 3, 5, 5, 6, 6, 7], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [Eflows["curtailed"], Eflows["E_electrolyzer"], Eflows["E_ren"], Eflows["E_f"], 
                Eflows["E_ren"], Eflows["D=1"], Eflows["D=1"], Eflows["losses"], 
                Eflows["HP"], Eflows["HP"], Eflows["H_exp"], Eflows["losses"], 
                Eflows["E_ren"], Eflows["E_ren"]],
      #target = [2, 3, 8, 10, 9, 12, 12, 4, 5, 6, 11, 4, 7, 9],
       value = [Ecurt, Ee, Edir, f, 
                0*Edir, Eren, f, P2H_losses, 
                HP, 0*HP, Hexp, H2P_losses, 
                EH2P, 0*EH2P],
       color = ['rgba(245,222,179,0.3)', "rgba(107,142,35,0.5)", "rgba(0,100,0,0.7)", "rgba(105,105,105, 0.7)", 
                "rgba(0,100,0,0.7)", "lightgray", "rgba(105,105,105, 0.7)", "rgba(107,142,35,0.5)", 
                "lightgray", "lightgray", "lightgray", "rgba(107,142,35,0.5)", 
                "rgba(128,128,0,0.7)", "rgba(128,128,0,0.7)"],
      

  ))])

#fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.update_layout(title_text="Full power system decarbonization", font_size=10)
fig.write_html("SANKEY.html")

#########################################333

#hydrogen export first 

etaH2P=0.5
etaP2H=0.7
r = 1.43
f = 0.0
Edir =1-f
Ecurt = 0.09
Ee= r - Edir - Ecurt 
P2H_losses = (1-etaP2H)*Ee
H = etaP2H*Ee
Hexp = 0.0
HP=H
H2P_losses = (1-etaH2P)*HP
EH2P = etaH2P * HP #adjust
Eren = Edir + EH2P

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 70,
      thickness = 10,
      line = dict(color = "gray", width = 0.0),
      #label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "H", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      #label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      color =  ["#C1CDCD", "rgb(107,142,35)", "rgb(245,222,179)", "rgb(107,142,35)", "rgb(107,142,35)", "#C1CDCD", "#C1CDCD", "rgba(128,128,0,0.7)", "rgba(0,100,0,0.7)", "#C1CDCD", "#C1CDCD", "#C1CDCD","#C1CDCD"]
    ),
    link = dict(    
      source = [Eflows["r"], Eflows["r"], Eflows["r"], Eflows["f"], 
                Eflows["E_dir"], Eflows["E_ren"], Eflows["E_f"], Eflows["E_electrolyzer"], 
                Eflows["E_electrolyzer"], Eflows["H"], Eflows["H"], Eflows["HP"], 
                Eflows["HP"], Eflows["E_H2P"]], # indices correspond to labels,
      #source = [0, 0, 0, 1, 8, 9, 10, 3, 3, 5, 5, 6, 6, 7], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [Eflows["curtailed"], Eflows["E_electrolyzer"], Eflows["E_ren"], Eflows["E_f"], 
                Eflows["E_ren"], Eflows["D=1"], Eflows["D=1"], Eflows["losses"], 
                Eflows["HP"], Eflows["HP"], Eflows["H_exp"], Eflows["losses"], 
                Eflows["E_ren"], Eflows["E_ren"]],
      #target = [2, 3, 8, 10, 9, 12, 12, 4, 5, 6, 11, 4, 7, 9],
       value = [Ecurt, Ee, Edir, f, 
                0*Edir, Eren, f, P2H_losses, 
                HP, 0*HP, Hexp, H2P_losses, 
                EH2P, 0*EH2P],
       color = ['rgba(245,222,179,0.3)', "rgba(107,142,35,0.5)", "rgba(0,100,0,0.7)", "rgba(105,105,105, 0.7)", 
                "rgba(0,100,0,0.7)", "lightgray", "rgba(105,105,105, 0.7)", "rgba(107,142,35,0.5)", 
                "lightgray", "lightgray", "lightgray", "rgba(107,142,35,0.5)", 
                "rgba(128,128,0,0.7)", "rgba(128,128,0,0.7)"],
      

  ))])

#fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.update_layout(title_text="Full power system decarbonization", font_size=10)
fig.write_html("SANKEY H first.html")



########################################################################
# =============================================================================
# 
# fig = go.Figure(data=[go.Sankey(
#     node = dict(
#       pad = 15,
#       thickness = 2,
#       line = dict(color = "black", width = 0.0),
#       label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "H", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
#       color = "black"# ["blue", "blue", "blue", "blue", "blue", "black"]
#     ),
#     link = dict(    
#       source = [Eflows["r"], Eflows["r"], Eflows["r"], Eflows["f"], 
#                 Eflows["E_dir"], Eflows["E_ren"], Eflows["E_f"], Eflows["E_electrolyzer"], 
#                 Eflows["E_electrolyzer"], Eflows["H"], Eflows["H"], Eflows["HP"], 
#                 Eflows["HP"], Eflows["E_H2P"]], # indices correspond to labels,
#       #source = [0, 0, 0, 1, 8, 9, 10, 3, 3, 5, 5, 6, 6, 7], # indices correspond to labels, eg A1, A2, A1, B1, ...
#       target = [Eflows["curtailed"], Eflows["E_electrolyzer"], Eflows["E_dir"], Eflows["E_f"], 
#                 Eflows["E_ren"], Eflows["D=1"], Eflows["D=1"], Eflows["losses"], 
#                 Eflows["H"], Eflows["HP"], Eflows["H_exp"], Eflows["losses"], 
#                 Eflows["E_H2P"], Eflows["E_ren"]],
#       #target = [2, 3, 8, 10, 9, 12, 12, 4, 5, 6, 11, 4, 7, 9],
#       value = [Ecurt, Ee, Edir, f, 
#                Edir, Eren, f, P2H_losses, 
#                H, HP, Hexp, H2P_losses, 
#                EH2P, EH2P],
#       color = ['rgba(245,222,179,0.3)', "rgba(107,142,35,0.5)", "rgba(0,100,0,0.7)", "rgba(105,105,105, 0.7)", 
#                "rgba(0,100,0,0.7)", "lightgray", "rgba(105,105,105, 0.7)", "rgba(107,142,35,0.5)", 
#                "lightgray", "lightgray", "lightgray", "rgba(107,142,35,0.5)", 
#                "rgba(128,128,0,0.7)", "rgba(128,128,0,0.7)"],
#       
# 
#   ))])
# =============================================================================
etaH2P=0.5
etaP2H=0.7
r = 1.43
f = 0.0
Edir =1-f
Ecurt = 0.09
Ee= r - Edir - Ecurt 
P2H_losses = (1-etaP2H)*Ee
H = etaP2H*Ee
Hexp = 0.0
HP=H
H2P_losses = (1-etaH2P)*HP
EH2P = etaH2P * HP #adjust
Eren = Edir + EH2P

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 50,
      thickness = 10,
      line = dict(color = "red", width = 0.0),
      label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "H", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      #label = ["r", "f", "curtailed", r'$E_{electrolyzer}$', "losses", "HP", "E_H2P", "E_dir", "E_ren", "E_f", "H_exp", "D=1"],
      color =  ["blue", "blue", "blue", "blue", "blue", "black", "gray", "rgba(128,128,0,0.7)", "rgba(0,100,0,0.7)", "yellow", "green"]
    ),
    link = dict(    
      source = [Eflows["r"], Eflows["r"], Eflows["r"], Eflows["f"], 
                Eflows["E_dir"], Eflows["E_ren"], Eflows["E_f"], Eflows["E_electrolyzer"], 
                Eflows["E_electrolyzer"], Eflows["H"], Eflows["H"], Eflows["HP"], 
                Eflows["HP"], Eflows["E_H2P"]], # indices correspond to labels,
      #source = [0, 0, 0, 1, 8, 9, 10, 3, 3, 5, 5, 6, 6, 7], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = [Eflows["curtailed"], Eflows["E_electrolyzer"], Eflows["E_dir"], Eflows["E_f"], 
                Eflows["E_ren"], Eflows["D=1"], Eflows["D=1"], Eflows["losses"], 
                Eflows["HP"], Eflows["HP"], Eflows["H_exp"], Eflows["losses"], 
                Eflows["E_H2P"], Eflows["E_ren"]],
      #target = [2, 3, 8, 10, 9, 12, 12, 4, 5, 6, 11, 4, 7, 9],
       value = [Ecurt, Ee, Edir, f, 
                Edir, Eren, f, P2H_losses, 
                HP, 0*HP, Hexp, H2P_losses, 
                EH2P, EH2P],
       color = ['rgba(245,222,179,0.3)', "rgba(107,142,35,0.5)", "rgba(0,100,0,0.7)", "rgba(105,105,105, 0.7)", 
                "rgba(0,100,0,0.7)", "lightgray", "rgba(105,105,105, 0.7)", "rgba(107,142,35,0.5)", 
                "lightgray", "lightgray", "lightgray", "rgba(107,142,35,0.5)", 
                "rgba(128,128,0,0.7)", "rgba(128,128,0,0.7)"],
      

  ))])

#fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.update_layout(title_text="Full power system decarbonization", font_size=10)
fig.write_html("SANKEY2.html")
