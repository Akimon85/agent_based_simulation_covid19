#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run
from mesa.visualization.UserParam import UserSettableParameter                                               
import scipy.stats as ss


# In[2]:


#Simulation Model

class Person(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.test_result = -1
        self.partied = 0
        self.serial_test = 0
        self.wait_result = -1
        self.exposed = -1
        first_infected = self.random.random()
        if first_infected < start_infection_rate:
            self.health = 1
            self.quarantine = 0
            self.d_infected = 1
            self.infected_from = "external"
            if self.random.random() <= 0.40:
                self.infection_type = "asymptomatic"
            else: 
                self.infection_type = "pre-symptomatic"
            
        else:
            self.health = 0
            self.quarantine = 0
            self.d_infected = 0
            self.infected_from = "N/A"
            self.infection_type = "N/A"
        
    def step(self):
        self.advance_infection()
        if self.wait_result > -1:
            self.wait_result =- 1
            if self.wait_result == 0:
                self.test_result == 1
        if self.serial_test > 0:
            self.serial_test =- 1
            if self.serial_test == 1:
                self.test()
        if self.exposed > -1:
            self.exposed =- 1
            if self.exposed == 0 and self.random.random() <= responsibility:
                self.test()
        if self.partied > 0:
            self.partied =- 1
            if self.partied == 1 and self.random.random() <= responsibility:
                self.test()
        if self.infection_type == "symptomatic" and self.random.random() <= responsibility and self.test_result == -1:
            self.test()
        if self.test_result == 1:
            if self.health == 1:
                self.quarantine = 1
            else:
                self.quarantine = 0
        if self.quarantine != 1 and self.health != 2:
            self.move()
        if self.health == 1 and self.d_infected > 1:
            self.infect()
        self.party()

    def infect(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for cellmate in cellmates:
            if cellmate.health == 0:
                if self.infection_type == "symptomatic":
                    cellmate.health = 1
                    cellmate.exposed = 2
                    cellmate.infected_from = self.infection_type
                    if self.random.random() <= infection_chance:
                        cellmate.infection_type = "asymptomatic"
                    else:
                        cellmate.infection_type = "pre-symptomatic"
                elif self.infection_type == "asymptomatic" and self.random.random() <= 0.63*infection_chance:
                    cellmate.health = 1
                    cellmate.infected_from = self.infection_type
                    if self.random.random() <= 0.40:
                        cellmate.infection_type = "asymptomatic"
                    else:
                        cellmate.infection_type = "pre-symptomatic"
                elif self.infection_type == "pre-symptomatic" and self.random.random() <= 0.45*infection_chance:
                    cellmate.health = 1
                    cellmate.infected_from = self.infection_type
                    if self.random.random() <= 0.40:
                        cellmate.infection_type = "asymptomatic"
                    else:
                        cellmate.infection_type = "pre-symptomatic"
                    
                    
                
    def advance_infection(self):
        if self.health == 1:
            self.d_infected += 1
            if self.infection_type == "pre-symptomatic" and self.d_infected >= self.random.randint(4,6):
                self.infection_type = "symptomatic"
            if self.d_infected == 17 and self.infection_type == "symptomatic":
                d_chance = self.random.random()
                if d_chance <= death_rate:
                    self.health = 2
            if self.d_infected == 18:
                self.health = 3
                self.infection_type = "N/A"

                    
            
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            if neighbor.quarantine == 1:
                possible_steps = [sub for sub in possible_steps if sub not in neighbor.pos]
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
    def test(self):
        if test == "antigen_rapid":
            if self.serial_test <= 1:
                if self.serial_test == 0:
                    self.serial_test = 3
                if self.health == 1:
                    if self.infection_type == "symptomatic":
                        if self.random.random() <= 0.642:
                            self.test_result = 1
                        else:
                            self.test_result = 0
                    elif self.infection_type == "asymptomatic" and self.d_infected >= 5:
                        if self.random.random() <= 0.355:
                            self.test_result = 1
                        else:
                            self.test_result = 0
        elif test == "molecular_rapid":
            if self.d_infected > 1 and self.random.random() <= 0.97:
                self.test_result = 1
            else:
                self.test_result = 0
        elif test == "molecular_lab":
            if self.d_infected > 1 and self.wait_result == -1:
                self.wait_result = 2
                
    def party(self):
        cell_content = self.model.grid.get_cell_list_contents([self.pos])
        agent_count = len(cell_content)
        if agent_count > 2 and self.partied > 0:
            self.partied = 3
    
class InfectionModel(Model):
    
    def __init__(self, N, width, height, start_infection_rate, test, infection_chance, responsibility):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        
        
        for i in range(self.num_agents):
            a = Person(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y)) 
    
        self.datacollector = DataCollector(
            #model_reporters={"Effective Reproduction Number": compute_r}, 
            agent_reporters={"Health Status": "health", 
                             "Infection Type": "infection_type",
                             "Quarantine Status": "quarantine",
                             "Infected From": "infected_from"}
        )
        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# In[3]:


#Single Test Run

infection_chance = 0.5
start_infection_rate = 0.01
symptomatic_rate = 0.66
death_rate = 0.02
responsibility = 1
test = "molecular_rapid"

model = InfectionModel(1500, 50, 50, start_infection_rate, test, infection_chance, responsibility)
for i in range(100):
    model.step()


# In[4]:


#Simulation Run and Visualization
model_params = {
    'N': UserSettableParameter(
        'number', 'Number of agents', 750, 50, 2000, 50),
    'width': 50,
    'height': 50,
    'start_infection_rate': UserSettableParameter(
        'slider', 'prop. of initial pop. infected', 0.03, 0, 0.2, 0.01),
    'test': UserSettableParameter(
        'choice', 'Test Type', value='molecular_rapid', choices=['antigen_rapid', 'molecular_rapid', 'molecular_lab']),
    'responsibility': UserSettableParameter(
        'slider', 'Agent Adherence To CDC Testing Guidelines', 0.5, 0, 1, 0.05),
    'infection_chance': UserSettableParameter(
        'slider', 'Infection Chance', 0.75, 0, 1, 0.05)
}




from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Layer": 0,
        "Color": "green",
        "r": 0.7}

    if agent.health == 1: 
        portrayal["Color"] = "red"
        portrayal["r"] = 0.7
    if agent.infection_type == "symptomatic":
        portrayal["Filled"] = 'true'  
    if agent.health == 2:
        portrayal["Color"] = "black"
        portrayal["r"] = 0.5
    if agent.health == 3:
        portrayal["Color"] = "blue"

    
    return portrayal

grid = CanvasGrid(agent_portrayal, 50, 50, 700, 700)

server = ModularServer(InfectionModel,
                       [grid],
                       "Infection Model",
                       model_params)
server.port = 6868 # The default
server.launch()


# In[ ]:


#Batch Runs

params = {"width": 50, "height": 50, 
          "N":[500, 1000],
          "start_infection_rate": 0.02 ,
          "test": {"antigen_rapid", "molecular_rapid", "molecular_lab"},
          "infection_chance": 1,
          "responsibility": [0.25,0.5,1]
         }

results = batch_run(
    InfectionModel,
    parameters = params,
    iterations = 5,
    number_processes = 1,
    max_steps = 100,
    data_collection_period=1,
    display_progress=True,
)


# In[ ]:


results_df = pd.DataFrame(results)


# In[ ]:


agent_infected = results_df[results_df['Health Status']==1].groupby(['RunId','iteration','Step','N', 'test', 'responsibility'])['Health Status'].count().reset_index()
agent_infected['# infected'] = agent_infected['Health Status']


# In[ ]:


sns.relplot(x="Step", y="# infected", hue="test", row="N", col="responsibility", kind="line", data=agent_infected.query("N == 1000"));


# In[ ]:


sns.relplot(x="Step", y="# infected", hue="test", row="N", col="responsibility", kind="line", data=agent_infected.query("N == 500"));



