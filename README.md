# ReAllY

Your Reinforcement Learning Ally.

A framework based on tensorflow, RAY and gym to handle reinforcement learning tasks.



## General Remarks

This framework was originally build for the block course "Deep Reinforcement Learning" at the University of Osnabrück.
In the block course students are asked to implement deep RL algorithms with the help of the framework. Sample solutions will be published once the course is done.

This framework is still under construction and can yet be optimized. If you run into errors or see ways to make something more efficient, please feel free to raise an issue or contact me dircetly (Charlie Lange, chalange@uos.de) and help make this framework better for everyone!

## Installation 
At best create a conda environment with the given environment.yml file / refer to the file for version specifications (important are: ray, gym, tensorflow, numpy, matplotlib, scipy, opencv).

Additionall install the really and gridworld package as following:

##### Really
```console
(ReAlly) yourname@device:~/your_path_to_really/ReAllY$ cd really
(ReAlly) yourname@device:~/your_path_to_really/ReAllY/really$ pip install -e . 
```
##### Gridworlds
```console
(ReAlly) yourname@device:~/your_path_to_really/ReAllY$ cd gridworlds
(ReAlly) yourname@device:~/your_path_to_really/ReAllY/gridworlds$ pip install -e . 
```
## General Design

-> insert graphic

### Sample Mnager
The sample manager manages collecting experience using remote runners. Therfore it has to be initialized with the specifications for the environment and the model used, how the agent should behave and what data needs to be collected. 

#### Buffer
Using the sample manager, a buffer can be initialized in which the data the sample manager collects via its remote runners can be stored and from which the user can sample data.

#### Aggregator 
Using the sample manager, an evaluation aggregator can be initialized where the training progress from the main process can be stored. This aggregator already computes smooth averages and handles plotting.


## Usage
### Model
The model **needs to output a dictionary** with either the keys 'q_values' (and optional 'v_estimate') or in case of policy optimization 'policy' or mu' and 'sigma' if it is a continous model. 

### Main Process
- ray needs to be initialized before the sample manager is used (ray.init(log_to_driver=False) to suppress logs)

### Sample Manager
The sample manager should be initalized from the main process.

#### Initialization

    @args:
        model: model Object, model: tf.keras.Model (or model imitating a tf model) returning dictionary with the possible keys: 'q_values' or 'policy' or 'mus' and 'sigmas' for continuous policies, optional 'value_estimate', containing tensors
        environment: string specifying gym environment or object of custom gym-like (implementing the same methods) environment
        num_parallel: int, number of how many agents to run in parall
        total_steps: int, how many steps to collect for the experience replay
        returns: list of strings specifying what is to be returned by the box
            supported are: 'value_estimate', 'log_prob', 'monte_carlo'
        actin_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', 'discrete_policy' or 'continuous_normal_diagonal'

    @kwargs:
        model_kwargs: dict, optional model initialization specifications
        weights: optional, weights which can be loaded into the agent for remote data collecting
        input_shape: shape or boolean (if shape not needed for first call of model), defaults shape of the environments reset state

        env_config: dict, opitonal configurations for environment creation if a custom environment is used

        num_episodes: specifies the total number of episodes to run on the environment for each runner, defaults to 1
        num_steps: specifies the total number of steps to run on the environment for each runner

        gamma: float, discount factor for monte carlo return, defaults to 0.99
        temperature: float, temperature for thomson sampling, defaults to 1
        epsilon: epsilon for epsilon greedy sampling, defaults to 0.95

        remote_min_returns: int, minimum number of remote runner results to wait for, defaults to 10% of num_parallel
        remote_time_out: float, maximum amount of time (in seconds) to wait on the remote runner results, defaults to None

        is_tf: boolean, if model is tensorflow model and neets initialization
 
##### Example:

    kwargs = {
        'model' : MyModel,
        'environment' :'CartPole-v0',
        'num_parallel' :5,
        'total_steps' :100,
        'returns' : ['monte_carlo']
        'action_sampling_type' :'thompson',
        'model_kwargs': {
            'num_actions: 2 }
        'temperature' : 0.5,
        'num_episodes': 20
    }

    manager = SampleManager(**kwargs)

#### Methods

##### Handling data
    manager.get_data()
        returns: dictionary with remoteley collected data according to the managers specifications

    manager.initilize_buffer(buffer_size, optim_keys=['state', 'action', 'reward', 'state_new', 'terminal'])
        args: 
            buffer_size: [int] size of the buffer
        kwargs:
            optim_keys: [list of strings] specifies the data to be stored in the buffer
         -> the buffer is a dictonary with a list for each of its keys with a max len of buffer_size
        
     manager.store_in_buffer(data)
        args:
            data: [dict] dict with list/np arrays/or tensors for each key
        -> when adding the new data leads to overstepping the buffer size, the data is handled according to the FIFO principle
        
     manager.sample(sample_size, from_buffer=True)
        args:
            sample_size: [int] how many samples to draw
         kwargs: 
            from_buffer: [boolean] if True samples are randomly drawn from the buffer, else new samples are created using the remote runners and the current agent
  
  ##### Handling the agent
  
     manager.get_agent()
      -> returns the current agent, an agent has the methods:
            agent.act(state)
            agent.max_q(state)
            agent.q_val(state, action)
            agent.get_weights()
            agent.set_weights(weights)
            
     manager.set_agent(weights)
        -> updates the current agents weights
    
    manager.save_model(path, epoch, model_name='model')
        args: 
            path: [str] specifying the relative path to the folder where the model should be saved in 
        -> creates a folder in the folder specified by 'path' where the current model is saved using tensorflows model.save (only supported for tf models currently!)
        
     manager.load_model(path)
        -> loads the most recent model in the folder specified by path using tf.keras.models.load_model()
        returns: agent with loaded weights
        
 ##### Handling specifications
     manager.set_temperature(temperature)
     
     manager.set_epsilon(epsilon)
     
     manager.set_env(env_kwargs)
 
        
 ##### Handling evaluation
      
      manager.test(self, max_steps, test_episodes=100, evaluation_measure='time', render=False, do_print=False)
        args: 
            max_steps: [int] how many max steps to take in the environment
         kwargs: 
            test_episodes: [int] how many episodes to test
            evaluation_measure: [string] specifiying what to be returned, can be 'time', 'reward' or 'time_and_reward'
            render: [boolean] whether to render the environmend while testing
            do_print: [boolean] whether to print the mean evaluation measure
          -> returns list of mean time steps or rewards or both
          -> when testing, all action sampling types will be set to greedy mode
        
      manager.initialize_aggregator(self, path, saving_after=10, aggregator_keys=['loss'])
        args:
            path: [str] relative path specifiying the folder where plots tracking the training progress should be saved to
        kwargs: 
            saving_after: [int] after how many iterations of aggregator updates the results should be visualized and saved in a plot
            aggregator_keys: [list of strings] keys for the internal dictonary specifiying what should be tracked
       
       manager.update_aggregator(**kwargs)
       -> takes a list of named arguments where the names should match the aggregators keys and refers to a list of values to be stored in the aggregator
       
 ##### Prebuilt agent functionalities
        agent.act(states)
            -> acts according to the policy, returns actions
        
        agent.v(states)
            -> returns the value estimate
        
        agent.q_val(states, actions)
        
        agent.max_q(states)
        
        agent.flowing_log_prob(states, actions, return_entropy=False)
            -> returns log probabilities with gradient flow 
            -> optional entropy
    
   All funcionalities are demonstrated in the showcase.py script.
    
    

