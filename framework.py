import system_dynamics
import networks
import numpy as np

# This parent class allows you to assemble networks as lego blocks.
# Only allowing 1D linear connections for now.
class Framework():
    def __init__(self):
        self.dyn_list = []
        self.net_list = []
    
    def add_net(self, net):
        # Adds a network to the framework
        self.net_list.apped(net)