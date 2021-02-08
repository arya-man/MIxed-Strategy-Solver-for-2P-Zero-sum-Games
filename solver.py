import numpy as np
import pulp as p

game = [[7,5,5],[5,4,5],[5,3,5]]
row =np.asarray(game)
col=-row
scaling = np.amax(row)
row = row + scaling
col = col + scaling
game_shape = list(np.shape(game))

#player1=['var_p1_' + str(i) for i in range(1,game_shape[0]+1)]
#player2=['var_p2_' + str(i) for i in range(1,game_shape[1]+1)]
#player1=np.asarray(player_1)
#player2=np.asarray(player_2)
#player_1=[p.LpVariable(i, lowBound = 0) for i in player_1] 
#player_1=player_1.reshape(1,game_shape[0])
#player_2=player_2.reshape(game_shape[1],1)

player_1=p.LpVariable.dicts('p1', range(0,game_shape[0]),lowBound=0)
player_2=p.LpVariable.dicts('p2',range(0,game_shape[1]),lowBound=0)

    
Lp_prob = p.LpProblem('Problem', p.LpMaximize)

sums=0

for i in range(0,game_shape[0]):
    sums+=player_1[i]
    
    
for j in range(0,game_shape[1]):
    sums+=player_2[j]
    
Lp_prob += sums

player1=[]
player2=[]

for i in range(0,game_shape[0]):
    player1.append(player_1[i])
    
for i in range(0,game_shape[1]):
    player2.append(player_2[i])
    

player1=np.asarray(player1)
player2=np.asarray(player2)

player1=player1.reshape(1,game_shape[0])
player2=player2.reshape(game_shape[1],1)

player2=np.matmul(row,player2)
player1=np.matmul(player1,col).T

p1_shape=list(player1.shape)
p2_shape=list(player2.shape)

for i in range(0,p2_shape[0]):
    Lp_prob += player2[i][0]<=1

    
for i in range(0,p1_shape[0]):
    Lp_prob += player1[i][0] <=1
                            
status = Lp_prob.solve() # Solver 

sums1=0
sums2=0
for i in range(0,game_shape[0]):
    sums1+= p.value(player_1[i])
    
for i in range(0,game_shape[1]):
    sums2+= p.value(player_2[i])
    
final1=[]
final2=[]

if sums1!=0:
    for i in range(0,game_shape[0]):
        final1.append(p.value(player_1[i])/sums1)
        
if sums2!=0:
    for i in range(0,game_shape[1]):
        final2.append(p.value(player_2[i])/sums2)
        
    
print(final1,final2)
