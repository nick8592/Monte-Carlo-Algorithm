# Monte-Carlo-Algorithm

## Task
1. Implement MC-Epsilon Greedy and compare it with MC-Exploring Start.
2. From s13 a drone can take our robot to s1, and from s9 same drone can take it to s6. (note that you need to add a new action called fly).

![map](https://user-images.githubusercontent.com/56616275/194511983-ba696c89-23ec-40d3-a241-d25349ac6fcf.png)

## Code Explain
### Implement MC-Epsilon Greedy - Monte_Carlo_Algorithm.py
The main difference between class `MC_Exploring_Start` and class `MC_Epsilon_Greedy` 
in my code is function `iter` part. 

After we use `generate_inital_state` and have the current state coordinate, 
I added an `if...else...` statement and parameter `EPSILON` to implement Epsilon Greedy concept. 
Parameter `EPSILON` defined as 0.1.
By using `random()` function to generate a float number between 0 and 1, 
for example let’s call it `rand_num`. 
 - If `rand_num` is **bigger** than the `EPSILON`,   
then remain with current action.  
 - If `rand_num` is **smaller** than the `EPSILON`,   
use `generate_random_action` randomly choose an action.  

And the final optimal policy also shown as below.  

![optimal_policy](https://user-images.githubusercontent.com/56616275/194511792-97ebd24f-d0cd-4dc3-93db-7b26dce6bc70.png)

By doing so, we can see the converged result from the figure shown below. 
Compare `MC Epsilon Greedy` with `MC Exploring Start`, 
MC Epsilon Greedy has 10% chance to randomly select an action, 
as you can see its line on chart greadually close to MC Exploring Start. 
If we decrease the EPSILON to 0.01, which means only 1% chance to randomly select an action, 
then MC Epsilon Greedy has less chance to take reandom action, which cause it need more iteration to converge.
Below shows when `EPSILON` equals to 0.01, 0.1, 0.5, 0.9.

![epsilon_0 01](https://user-images.githubusercontent.com/56616275/194511577-cd362bf2-3301-42d2-91a9-0636018f6592.png)
![epsilon_0 1](https://user-images.githubusercontent.com/56616275/194511663-5b34b813-feb1-400a-89ac-d7c9c19673c9.png)
![epsilon_0 5](https://user-images.githubusercontent.com/56616275/194511708-ff8f2a1e-f307-445e-b412-7cd75a3d97f1.png)
![epsilon_0 9](https://user-images.githubusercontent.com/56616275/194511734-114e64ee-512f-4ac0-92f8-f6703913238b.png)

### Add drone - MC_with_drone.py
First, added `fly` action into action dictionary, 
with `[0, 0]` value (later will give it different value based on different state, s9 or s13).  

Second, at function `transfer_state`, I determined whether the input `state_coordinate` is 
`s9[1, 3]` or `s13[2, 4]` or other state coordinate.   

If `state_coordinate` is `s9[1, 3]`, then see what’s the random input action are given. 
If the random action is `fly`, then the drone will take the robot from `s9[1, 3]` to `s1[1, 0]`, 
which means robot need to stay on the same row but go left for 3 columns. 
As the code shown below, `state_coordinates + [0, -3]`. Otherwise, 
if the random action is not `fly` but other actions (up, down, right, left), 
then just add the moving action value according to the direction dictionary.   

Same concept for s13, if the input `state_coordinate` is `s13[2, 4]`, 
and the random action is `fly`. Then, the robot will move from `s13[2, 4]` to `s1[0, 1]`, 
which means the robot need to go up for 1 row and go left for 3 columns. 
As the code shown below, `state_coordinates + [-2, -3]`. 
Otherwise, if the random action is not `fly` but other action (up, down, right, left), 
then just add the moving action value according to the direction dictionary.   

And if the input `state_coordinate` is neither `s9` nor `s13`, 
then it must be other state in range `s1`, `s2`, ..., `s8`, `s10`, `s11`, `s12`, `s14`, ..., `s20`. 
In this case, just add the moving action value according to the direction dictionary. 
By doing so, we can calculate the sum of `state_coordinate` and `action value`, 
and get the `next state coordinate`.  

After getting the coordinate for the next state, 
we determine whether `next state coordinate` is out of the board or not. 
Range from 0 to 3 for row, and 0 to 5 for column. 
If `next state coordinate` is out the board, 
then just return the value of the input `state coordinate` as we assign as `current_state_coordinate` at the very beginning.  

At last, we can assure that the `next state coordinate` is in the grid, 
then we need to check whether the next state is a `wall` or not. 
If next state is `wall`, apparently we can not choose it as our next step, 
therefore just return the `current_state_coordinate`.   

The optimal policy is shown below, at state `s9[1, 3]` and `s13[2, 4]` both will take the `fly` action.
Even `s10[1, 5]` and `s18[3, 3]` also influence by `s13`, 
both of the actions are try to go to state s13 for shorter route.   

![optimal_policy_with_drone](https://user-images.githubusercontent.com/56616275/194511846-2c4d3a97-2ecf-4304-ab5e-8e1aa2480290.png)
