# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np 
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.spatial.distance import __all__ as implemented_metrics

from ase.data import atomic_numbers

from ibslib import Structure
from ibslib.driver import BaseDriver_
from ibslib.io import read,write


import matplotlib.pyplot as plt


class BondNeighborhood(BaseDriver_):
    """
    Returns the bonding neighborhood of each atom for a structure. User is 
    allowed to define a radius that the algorithm traverses to build 
    the neighborhood for each atom. If the radius is 0, this would 
    correspond to just return the atoms im the system.
    
    Arguments
    ---------
    radius: int
        Radius is the number of edges the model is allowed to traverse on the 
        graph representation of the molecule. 
    bond_kw: dict
        Keyword arguments for the molecule bonding module. The default values
        are the recommended settings. A user may potentially want to decrease
        the mult. This value is multiplied by covalent bond 
        radii in the MoleculeBonding class. It's highly recommended that the
        skin value is kept at 0.
        
    """
    def __init__(self, radius=1, 
                   bond_kw={"mult": 1.20,
                            "skin": 0,
                            "update": False},
                    cycles=False,
                ):
        self.radius = radius
        self.bond_kw = bond_kw
        self.cycles=cycles
        if radius != 1:
            raise Exception("Radius greater than 1 not implemented")
        
    
    def calc_struct(self, struct):
        self.struct = struct
        self.ele = struct.geometry["element"]
        g = self._build_graph(struct)
        n = self._calc_neighbors(g)
        n = self._sort(g,n)
        fragments = self._construct_fragment_list(n)
        fragments,count = np.unique(fragments, return_counts=True)
        fragments = list(fragments)
        count = [int(x) for x in count]
        
        if self.cycles:
            cycles_frag,cycle_count = self._calc_cycles(g)
            for cycle_name in cycles_frag:
                fragments.append(cycle_name)
            for value in cycle_count:
                count.append(int(value))
        
        self.struct.properties["bond_neighborhood"] = [fragments,
                    count]
        self.struct.properties["bond_neighborhood_fragments"] = fragments
        self.struct.properties["bond_neighborhood_counts"] = count
        
        return fragments,count
        
    
    def _build_graph(self, struct):
        """
        Builds networkx graph of a structures bonding.
        """
        
        bonds = struct.get_bonds(**self.bond_kw)
        g = nx.Graph()
        
        # Add node to graph for each atom in struct
        self.ele = struct.geometry["element"]
        g.add_nodes_from(range(len(self.ele)))
        
        # Add edges
        for i,bond_list in enumerate(bonds):
            [g.add_edge(i,x) for x in bond_list]
        
        return g
    
    
    def _calc_neighbors(self, g):
        """
        Calculates neighbors for each node in the graph. Uses the radius 
        which was declared when BondNeighborhood was initialized. Ordering
        of the neighbors follows:
            1. Terminal atom alphabetical 
            2. Self
            3. Bonded atom alphabetical
            4. If continuing from bonded group, print terminal groups of the 
               bonded group. 
            5. If there's ambiguity about which should come next, place in 
               alphabetical order. When the radius is small, there's less
               ambiguity. If the radius becomes large there will be more. 
               Although, only a radius of 1 is currently implemented.
            
        """
        neighbors = [[[]] for x in g.nodes]
        
        for i,idx_list in enumerate(neighbors):
            neighbor_list = [x for x in g.adj[i]]
            neighbor_list.append(i)
            idx_list[0] += neighbor_list
        
        return neighbors
    
    
    def _calc_cycles(self, g, neighborlist=[]):
        """
        Calculates the cycles in the graph. 
        
        Cycles are sorted in the following way:
            1. Start with the heaviest atom
            2. If there is more than one heaviest atom, begin with the one 
               bonded to the next heaviest. 
        
        """
        if type(g) == Structure:
            g = self._build_graph(g)
        
        if len(neighborlist) == 0:
            neighborlist = self._calc_neighbors(g)
        
        cycles = nx.cycles.cycle_basis(g)
        cycle_ele = []
        for cycle_idx in cycles:
            ele = self.ele[cycle_idx]
            z = [atomic_numbers[x] for x in ele]
            number,count = np.unique(z, return_counts=True) 
            
            max_number_idx = np.argmax(number)
            max_number = number[max_number_idx]
            max_number_count = count[max_number_idx]
            
            ## Have more than one heavy atom
            if max_number_count > 1:
                max_idx = np.where(z == max_number)[0]
                
                ## Find which one is bonded to heaviest element
                chosen_idx = max_idx[0]
                max_neighbor = 0
                for idx in max_idx:
                    neighbors = neighborlist[idx]
                    neighbor_ele = self.ele[neighbors]
                    z = [atomic_numbers[x] for x in neighbor_ele]
                    temp_max_neighbor = np.max(z)
                    if temp_max_neighbor > max_neighbor:
                        max_neighbor = temp_max_neighbor
                        chosen_idx = idx
            else:
                chosen_idx = np.where(z == max_number)[0][0]
            
            ### Now build cycle ordering
            temp_cycle_ele = [ele[chosen_idx]]
            current_idx = chosen_idx
            for i in range(len(cycle_idx)-1):
                current_idx += 1
                if current_idx == len(cycle_idx):
                    current_idx -= len(cycle_idx)
                temp_cycle_ele.append(ele[current_idx])
            
            cycle_ele.append(temp_cycle_ele)
        
        unique_cycles = []
        counts = []
        for idx1,cycle_list1 in enumerate(cycle_ele):
            
            if cycle_list1 in unique_cycles:
                continue
            
            unique_cycles.append(cycle_list1)
            count = 1
            
            for idx2,cycle_list2 in enumerate(cycle_ele[idx1+1:]):
                idx2 += idx1+1
                if cycle_list1 == cycle_list2:
                    count += 1
            
            counts.append(count)
        
        frag_name_list = []
        for cycle_list in unique_cycles:
            temp_name = "cycle_"
            temp_name += "".join(cycle_list)
            frag_name_list.append(temp_name)
            
        return frag_name_list,counts
    
    
    def _sort(self, g, neighbor_list):
        """
        Sorts neighborlist according to definition in _calc_neighbors. Only 
        works for a radius of 1.
        
        Arguments
        ---------
        g: nx.Graph
        neighbor_list: list of int
            List of adjacent nodes plus the node itself as i
        
        """
        sorted_list_final = [[[]] for x in g.nodes]
        for i,temp in enumerate(neighbor_list):
            # Preparing things which aren't writting well 
            idx_list = temp[0]
            current_node = i
            
            terminal_groups = []
            bonded_groups = []
            for idx in idx_list:
                if g.degree(idx) == 1:
                    terminal_groups.append(idx)
                else:
                    bonded_groups.append(idx)
            
            terminal_ele = self.ele[terminal_groups]
            alphabet_idx = np.argsort(terminal_ele)
            terminal_groups = [terminal_groups[x] for x in alphabet_idx]
            
            sorted_list = terminal_groups
            if current_node not in terminal_groups:
                sorted_list.append(current_node)
                remove_idx = bonded_groups.index(current_node)
                del(bonded_groups[remove_idx])
            
            bonded_ele = self.ele[bonded_groups]
            alphabet_idx = np.argsort(bonded_ele)
            bonded_groups = [bonded_groups[x] for x in alphabet_idx]
            
            sorted_list += bonded_groups
            
            sorted_list_final[i][0] = sorted_list
        
        return sorted_list_final
    
    
    def _construct_fragment_list(self,n):
        fragment_list = [self.struct.geometry["element"][tuple(x)] for x in n]
        fragment_list = [''.join(x) for x in fragment_list]
        return fragment_list


class construct_bond_neighborhood_model():
    """
    Manages the construction of a regularized bond neighborhood model. Also
    has many functions which can be used to perform analysis of the learned
    model or on the dataset which was used, such as computing similarity of 
    molecules across the dataset. Models can be trained using any user defined 
    target value. Current use case is for the prediction of solid form volumes 
    of molecules from the Cambridge Structural Database. 
    
    Usage
    -----
    0. Initialize a BondNeighborhood class. Initialized the constructor
       class with the BondNeighborhood and the list of extra properties 
       the user would like to include in the model as add_prop and the
       target property as target_prop.
    1. Call set_train with a Structure dictionary to construct the 
       neighborhood_df and target_df
    2. Call regularize to reduce the number of neighborhoods in the model.
    3-1. Can call get_feature_vector(struct) to get the feature vector for the 
         input structure which is correctly ordered with respect to the 
         regularized neighborhood dataframe. 
    3-2. Alternatively, can call set_test to build a neighborhood_df for a 
         test dataset and the target_df_t for the test dataset. 
         This neighborhood_df will have the same columns as the neighborhood_df 
         constructed by set_train.
    4. Call fit_model with a regressor object. This will return a dataframe 
       for the training set and testing set which contains the target values 
       and the predicted values. 
       
     
    Model Analysis
    --------------
    Functions which perform model analysis.
    
    Arguments
    ---------
    bn: BondNeighborhood object
        Initialized BondNeighborhood object.
    add_prop: list of str
        List of properties stored in the structures which the user wants to 
        add to the feature set.
    target_prop: list of str
        List of properties stored in the structures which the user wants to 
        use as the target values.
        
        
    """
    def __init__(self, bn, add_prop=["MC_volume"], target_prop=["molecule_volume"]):
        self.bn = bn
        self.add_prop = add_prop
        self.target_prop = target_prop
        
        ##### Initialize internals
        self.neighborhood_df = pd.DataFrame()
        self.neighborhood_df_r = pd.DataFrame() # Regularized neighborhood df
        self.total_neighborhoods_observed = 0
        # neighborhood df built for testing dataset populated by set_test
        self.neighborhood_df_t = pd.DataFrame() 
        
        self.target_df = pd.DataFrame() # Target df for training set
        self.target_df_t = pd.DataFrame() # Target df to testing set
        
        
    def fit_model(self, regressor, r=True, test=True):
        """
        Fits and tests the model. Regressor object must have a regressor.fit 
        method. 
        
        Arguments
        ---------
        regressor: object
            Regressor object which has a regressor.fit method. This can be
            any regressor from sklearn.
        r: bool
            True: The regularized self.neighborhood_df_r will be used
            False: The nonregularized self.neighborhood_df will be used
        test: bool
            True: Use the testing dataset
            False: Do not use the testing dataset
            
        """
        
        self._check_r_setting(r,"fit_model")
        
        # Choose training df based on r argument
        if r == True:
            temp_df = self.neighborhood_df_r
        else:
            temp_df = self.neighborhood_df
        
        if test and temp_df.shape[1] != self.neighborhood_df_t.shape[1]:
            raise Exception("The number of features of the training dataset"+
                    "is {} ".format(temp_df.shape[1]) + 
                    "which does not match the number of features "+
                    "in the testing dataset, which is {}."
                    .format(self.neighborhood_df_t.shape[1]))
            
        regressor.fit(temp_df.values, self.target_df.values)
        train_pred = regressor.predict(temp_df.values)
        
        #### Construct dataframe to return for the training dataframe
        pred_columns = ["predicted_"+ x for x in 
                        self.target_df.columns.tolist()]
        df_columns = self.target_df.columns.tolist() + pred_columns
        train_df = pd.DataFrame(data=np.zeros((self.target_df.shape[0],
                                               self.target_df.shape[1]*2)),
                                index=self.target_df.index,
                                columns=df_columns)
        
        self.train_df = train_df
        
        train_df.iloc[:,0:self.target_df.shape[1]] = self.target_df.values
        train_df.iloc[:,self.target_df.shape[1]:] = train_pred
        
        #### If using a testing dataset, calculate and construct dataframe
        if test:
            test_pred = regressor.predict(self.neighborhood_df_t.values)
            
            # Constructing test dataframe
            test_df = pd.DataFrame(data=np.zeros((self.target_df_t.shape[0],
                                               self.target_df_t.shape[1]*2)),
                                index=self.target_df_t.index,
                                columns=df_columns)
            test_df.iloc[:,0:self.target_df_t.shape[1]] = self.target_df_t.values
            test_df.iloc[:,self.target_df_t.shape[1]:] = test_pred
            
            return train_df,test_df
        else:
            return train_df
         
    
    def set_train(self, struct_dict):
        """
        Set training struct dict and builds an internal neighborhood dataframe.
        First collects all fragments from the dataset. Then builds a dataframe
        using only the unique fragments observed from the dataset. 
        
        Arugments
        ---------
        struct_dict: StructDict
            Structure dictionary for which to train the model on.
        add_prop: list of str
            List of any additional properties to add to the features set.
            
        """
        # Reinitialized internals when set_train is called
        self.neighborhood_df = pd.DataFrame()
        self.total_neighborhoods_observed = 0
        self.neighborhood_df_r = pd.DataFrame()
        
        # First get fragment and counts for each structure
        temp_columns = ["fragments", "counts"]
        temp_columns += self.add_prop
        struct_keys = [x for x in struct_dict.keys()]
        temp_df = pd.DataFrame(data=np.zeros((len(struct_dict),
                                              2+len(self.add_prop)),
                                             dtype=object),
                               index=struct_keys,
                               columns=temp_columns)
        # Store properties added for future reference neighborhood_df construction
        temp_add_prop_dict = {}
        
        for struct_id,struct in struct_dict.items():
            f,c = self.bn.calc_struct(struct)
            struct.set_property("bond_neighborhood_fragments", f)
            struct.set_property("bond_neighborhood_counts", c)
            
            values = [np.array(f),c]
            if len(self.add_prop) > 0:
                prop_values = self._get_add_prop(struct)
                temp_add_prop_dict[struct_id] = prop_values
                values += prop_values
            
            temp_df.loc[struct_id] = values
        
        # Building final neighborhood dataframe from unique bond neighborhoods
        all_fragments = np.hstack(temp_df["fragments"].values)
        unique_fragments = np.sort(np.unique(all_fragments))
        temp_columns = np.array([x for x in self.add_prop], dtype=object)
        temp_columns = np.hstack((temp_columns, unique_fragments))
        
        self.neighborhood_df = pd.DataFrame(
                                data=np.zeros((len(struct_dict),
                                               len(temp_columns))),
                                index = struct_keys,
                                columns = temp_columns)
        
        # Populate neighborhood_df                       
        for struct_id,struct in struct_dict.items():
            f = temp_df.loc[struct_id, "fragments"]
            c = temp_df.loc[struct_id, "counts"]
#            f = struct.properties["bond_neighborhood_fragments"]
#            c = struct.properties["bond_neighborhood_counts"]
            self.neighborhood_df.loc[struct_id, f] = c
            
            if len(self.add_prop) > 0:
                self.neighborhood_df.loc[struct_id, self.add_prop] = \
                                            temp_add_prop_dict[struct_id]
            
            # Updating the neighborhoods observed 
            self.total_neighborhoods_observed += np.sum(c)
        
        # Get target df for training
        self.target_df = self._get_target_df(struct_dict)
    
    
    def set_test(self, struct_dict, r=True):
        """
        Creates a neighborhood dataframe for the struct_dict which has the same
        columns as the internal neighborhood_df or neighborhood_df_r (the
        regularized dataframe). 
        
        Arugments
        ---------
        struct_dict: StructDict
            Dictionary indexed by structure ids with Structure obejct as the 
            values. 
        r: bool
            True: self.neighborhood_df_r will be used
            False: self.neighborhood_df will be used
            
        """
        # Check if self.neighborhood_df_r has been constructed
        self._check_r_setting(r,"set_test")
                
        if r == False:
            temp_df = self.neighborhood_df
        else:
            temp_df = self.neighborhood_df_r
            
        
        self.neighborhood_df_t = pd.DataFrame(
                 data = np.zeros((len(struct_dict),
                                  temp_df.columns.shape[0])),
                 index = [x for x in struct_dict.keys()],
                 columns = temp_df.columns.values)
        
        # Populate testing dataframe
        for struct_id,struct in struct_dict.items():
            nv = self.get_neighborhood_vector(struct,r=r)
            self.neighborhood_df_t.loc[struct_id] = nv
        
        # Populate target dataframe for test set
        self.target_df_t = self._get_target_df(struct_dict)
        
            
    def regularize(self, tol=10, unique=True):
        """
        Reduces the neighborhood_df to only those neighborhoods which have 
        been observed >= tol number of times in the training dataset.
        
        Arguments
        ---------
        tol: int or float
            If int, a neighborhood must be observed more than tol time in order 
            for it to be kept in the neighborhood_df_r.
            If float, a neighborhood must make up a this fraction of all 
            observed neighborhoods. This tolerance value should be extremely
            small such as 0.001 or 0.0001.
        unique: bool
            If True, tol is used as the number of structures for which the 
            neighborhood must be present in. 
            If False, then just the sum total of the times the neighborhood is
            observed is used. 
        
        """
        if len(self.neighborhood_df) == 0:
            raise Exception("Please call "+
                    "construct_bond_neighborhood_model.set_train before you "+
                    "call regularize.")
        drop_list = []
        for name in self.neighborhood_df.columns:
            
            if unique == False:
                # Find where columns are nonzero
                non_zero = np.sum(self.neighborhood_df[name] > 0) 
            else:
                non_zero = np.where(self.neighborhood_df[name] != 0)[0]
                non_zero = len(non_zero)
            
            if type(tol) == int:
                if non_zero < tol:
                    drop_list.append(name)
            elif type(tol) == float:
                if non_zero / self.total_neighborhoods_observed < tol:
                    drop_list.append(name)
        
        self.neighborhood_df_r = self.neighborhood_df.drop(drop_list,axis=1)
    
    
    def model_complexity_analysis(self, regressor, choose_n_features=-1,
                                  cross_validation=0,
                                  neighborhood_df_train=pd.DataFrame(),
                                  target_train = pd.DataFrame(),
                                  neighborhood_df_test=pd.DataFrame(),
                                  target_test = pd.DataFrame()):
        """
        Generates results testing how the training and testing error
        change as a function of the number of features included in the model, 
        otherwise called the model complexity. Idea here is that as model
        complexity increases, the training error will decrease monoatomically,
        however, at some point, the testing error should increase indicating 
        overfitting of the model. Features are added to the model in a forward,
        greedy way. 
        
        Arguments
        ---------
        regressor: object
            Supervised learning estimator with a fit method that provides
            information about the feature importance either through a coef_ or
            feature_importances_ attribute. 
        cross_validation: int
            Number of cross validation calculations to be performed for each
            point of the model complexity testing. If set to 0 or 1, no
            cross validation will be used. If greater than 1, then a standard
            deviation of the predicted error can be computed. 
        choose_n_features: 
            Number of features which will be chosen greedily. 
        neighborhood_df_train/test: pd.DataFrame
            Allows users to pass in their desired dataframe. Otherwise, resorts
            to the default behavior, which is to use the dataframes stored in 
            self.neighborhood_df/self.neighborhood_df_t for train/test.
            
        """
        if len(neighborhood_df_train) == 0:
            if len(self.neighborhood_df) == 0:
                raise Exception("Called constructor.model_complexity_analysis "+
                        "without setting a training dataset.")
            neighborhood_df_train = self.neighborhood_df
            target_train = self.target_df.values
        else:
            pass
        if len(neighborhood_df_test) == 0:
            if len(self.neighborhood_df_t) == 0:
                raise Exception("Called constructor.model_complexity_analysis "+
                        "without setting a testing dataset." +
                        "Note that one could use the same training and testing "+
                        "dataset by calling set_train with the trainig set.")
            neighborhood_df_test = self.neighborhood_df_t
            target_test = self.target_df_t
        else:
            pass
        
        # Defines default behavior for choose_n_features
        if choose_n_features <= 0:
            choose_n_features = len(neighborhood_df_train)
        if choose_n_features > len(neighborhood_df_train):
            choose_n_features = len(neighborhood_df_train)
            
        # Build greedy feature model
        greedy_features = pd.DataFrame()
        greedy_model_err = np.zeros((choose_n_features,2))
        train = neighborhood_df_train
        target = target_train
        test = neighborhood_df_test
        target_t = target_test.values
        for n in range(0,choose_n_features):
            # Reset greedy values to begin process of adding a new feature
            model_err = []
            if n == 0:
                greedy_values = np.zeros((len(train.index),1))
            else:
                # Add column for new feature
                greedy_values = np.zeros((greedy_features.values.shape[0],
                                          greedy_features.values.shape[1]+1))
                greedy_values[:,0:-1] = greedy_features.values
                
            for j,feature in enumerate(train.columns):
                print(n, j,len(train.columns),feature)
                train_temp = train.iloc[:,j].values
                
                # Add new feature from training set to greedy values
                greedy_values[:,-1] = train_temp
                
                regressor.fit(greedy_values,target)
                pred = regressor.predict(greedy_values)
                err = np.mean(np.abs(target - pred) / pred)
                model_err.append(err)
            
            # Choose best new feature to add
            best_new_feature_idx = np.argmin(model_err)
            feature_name = train.columns[best_new_feature_idx]
            feature_values = train[feature_name].values
            
            # Remove this feature from future consideration in the train set
            train = train.drop(feature_name,axis=1)
            
            if n == 0:
                greedy_features = pd.DataFrame(data=feature_values,
                                               index=train.index,
                                               columns=[feature_name])
            else:
                greedy_features[feature_name] = feature_values
            
            # Now use testing set with the new greedy features
            regressor.fit(greedy_features.values,target)
            pred_t = regressor.predict(test.loc[:,greedy_features.columns])
            err_t = np.mean(np.abs(target_t - pred_t) / pred_t)
            
            # Store error
            greedy_model_err[n,0] = model_err[best_new_feature_idx]
            greedy_model_err[n,1] = err_t
            
            print(greedy_model_err[n,:])
            
        return greedy_features,greedy_model_err
        
    
    
    def get_neighborhood_vector(self, struct, r=True):
        """
        Returns the neighborhood vector commensurate for either the regularized 
        neighborhood dataframe or the neighborhood dataframe from set_train.
        
        """
        # Check if self.neighborhood_df_r has been constructed
        self._check_r_setting(r,"get_neighborhood_vector")
                
        if r == False:
            temp_df = self.neighborhood_df
        else:
            temp_df = self.neighborhood_df_r
        
        fragment_list = temp_df.columns.values
        
        # Initialize feature vector for structure
        neighborhood_vector = np.zeros(fragment_list.shape)
        
        # Calc structure fragments and counts
        f,c = self.bn.calc_struct(struct)
        f = np.array(f)
        c = np.array(c)
        # Check which fragments in struct are in fragment list and return
        # the index of their location
        f_idx,c_idx = np.nonzero(fragment_list[:,None] == f)
        
        # Populate feature vector only with neighborhoods observed in temp_df
        neighborhood_vector[f_idx] = c[c_idx]
        
        #### Finish with add_prop values
        prop_values = self._get_add_prop(struct)
        p_idx = np.nonzero(fragment_list[:,None] == self.add_prop)[0]
        neighborhood_vector[p_idx] = prop_values
        
        return neighborhood_vector
    
    
    def _get_add_prop(self, struct):
        """
        Small function for obtaining the add_prop values of the input 
        structure. 
        
        """
        prop_values = []
        for prop_name in self.add_prop:
            temp_prop_value = struct.get_property(prop_name)
            
            # Define behavior of not finding the property
            if not temp_prop_value:
                temp_prop_value = 0
                    
            prop_values.append(temp_prop_value)
        return prop_values
    
    
    def _get_target_df(self, struct_dict):
        """
        Returns dataframe of the target properties for the struct_dict
        
        """
        target_df = pd.DataFrame(
                     data=np.zeros((len(struct_dict), len(self.target_prop))),
                     columns=self.target_prop,
                     index=[x for x in struct_dict.keys()])
        
        for struct_id,struct in struct_dict.items():
            prop_values = []
            for prop_name in self.target_prop:
                temp_prop_value = struct.get_property(prop_name)
            
                # Define behavior of not finding the property
                if not temp_prop_value:
                    temp_prop_value = 0
                
                prop_values.append(temp_prop_value)
                
            target_df.loc[struct_id] = prop_values
        
        return target_df
    
    
    def _check_r_setting(self,r,method_name):
         # Check if self.neighborhood_df_r has been constructed
        if len(self.neighborhood_df_r) == 0:
            if len(self.neighborhood_df) == 0:
                raise Exception("The user called "+
                   "construct_bond_neighborhood_model.{} ".format(method_name) +
                   "before calling set_train. " +
                   "Please call set_train, and " +
                   "optionally regularize, before calling get_feature_vector.")
            elif r == True:
                raise Exception("The user called "+
                    "construct_bond_neighborhood_model.{} ".format(method_name) +
                    "with r=True before calling regularize. " +
                    "Either call regularize prior to get_neighborhood_vector "+
                    "or set r=False.")
    
    
    def plot_results(self,result_df):
        """
        Quickly plots the results from the constructed model for the results
        dataframe.
        """
        pass
    
    
    def plot_hist(self, neighborhood_df, 
                  exclude_add_prop=True, 
                  figname='',
                  regressor=None, 
                  most_important=-1, 
                  add_coef=False,
                  add_coef_text_kwargs =  {
                                            "whitespace_pad": 1.15,
                                            "labelpad_y": 10,
                                            "labelpad_x": -0.065,
                                            "fontsize": 16,
                                            "color": "tab:red",                                            
                                          },
                  add_obs=False,
                  add_obs_text_kwargs =   {
                                            "whitespace_pad": 1.15,
                                            "labelpad_y": 35,
                                            "labelpad_x": -0.075,
                                            "fontsize": 16,
                                            "color": "tab:purple"
                                          },
                  figsize=(12,8),
                  tick_params_kwargs_both={
                                             "axis": "both",
                                             "width": 3,
                                             "labelsize": 16,
                                             "length": 8,
                                           },
                  tick_params_kwargs_x =  {
                                             "axis": "x",
                                             "labelsize": 16,
                                             "labelrotation": 90,
                                          },
                  y_label_kwargs =        {
                                             "ylabel": "Number of Observations",
                                             "labelpad": 0,
                                             "fontsize": 18,
                                             "labelpad": 15,
                                          },
                  bar_kwargs =            {
                                            "edgecolor": "k",
                                            "linewidth": 2,
                                          },
                  ):
        """
        Plots a histogram of the frequency of bond neighborhoods identified
        in the input neighborhood dataframe
        
        Arguments
        ---------
        neighborhood_df: pandas.DataFrame
            Neighborhood dataframe from which to construct the histogram.
        exclude_add_prop: bool
            Whether to use columns from add prop in the histogram
        figname: str
            If a figname is provided, then the file is saved using this 
            string.
        Regressor: regressor object
            If a regressor is provided, then the histogram is constructed using
            only the N first most important bond neighborhoods.
        most_important: int
            If a values greater than 0 is provided, then the regressor must 
            also be provide. Uses on the N most_important features in 
            construction of historgram if greater than 0.
        add_coef: bool
            Adds the coefficient values from the regressor to the top of each
            bar plot.
        add_coef_text_kwargs: dict
            Keyword arguments for the ax.text call for the coeficient values.
        add_obs: bool
            Adds the number of observations for each plotted neighborhood
            to the top of each bar plot.
        add_obs_text_kwargs: dict
            Keyword arguments for the ax.text call for the observation values.
        figsize: (int,int)
            Figure size to pass into matplotlib.
        tick_params_kwargs_both: dict
            Dictionary of keyword arguments to pass to ax.tick_params
        tick_params_kwargs_both: dict
            Another dictionary of keyword arguments to pass to ax.tick_params.
            Idea here is that the *_both is used to format both axes while 
            this dictionary is used to format the x axis.
        y_label_kwargs: dict
            Dictionary of keyword arguments to control the label for the 
            y axis of the plot. 
        bar_kwargs: dict
            Dictionary of keyword arguments to control aesthetics of the bar
            plot such as color, opacity, and linewidth.
        
        """
        if exclude_add_prop:
            column_idx = np.arange(len(self.add_prop), 
                                   len(neighborhood_df.columns),
                                   1)
        else:
            column_idx = np.arange(0,
                                   len(neighborhood_df.columns),
                                   1)
        
        if most_important > 0:
            if not regressor:
                raise Exception("Argument most_important to plot_hist was {}."
                        .format(most_important) +
                        "However a regressor was not provided. Please provide "+
                        "A trained regressor object from sklearn.")
            
            if len(regressor.coef_.ravel()) != len(neighborhood_df.columns.values):
                raise Exception("The length of the regressor coeficients was {} "
                        .format(len(regressor.coef_)) +
                        "which does not match the number of columns of the "+
                        "the input neighborhood dataframe {}"
                        .format(len(neighborhood_df.columns.values)))
                
            coef = np.abs(regressor.coef_.ravel()[column_idx])
            important_idx = np.argsort(coef)[::-1]
            important_idx = important_idx[0:most_important]
            
            # Now reset column_idx to only be these most important indices
            column_idx = column_idx[important_idx]
                       
        column_names = neighborhood_df.columns.values[column_idx]
        sub_array = neighborhood_df.values[:,column_idx]
        column_totals = np.sum(sub_array, axis=0)
        
        sort_idx = np.argsort(column_totals)[::-1]
        column_idx = column_idx[sort_idx]
        column_totals = column_totals[sort_idx]
        column_names = column_names[sort_idx]
        x = np.arange(0,len(column_names),1)
        
        #### Making figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.bar(x, column_totals, **bar_kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(column_names)
        ## Edit ticks
        ax.tick_params(**tick_params_kwargs_both)
        ax.tick_params(**tick_params_kwargs_x)
        
        # ylabel kwargs
        ax.set_ylabel(**y_label_kwargs)
        
        # Set outside of plot same size as tick_width
        outline_width = tick_params_kwargs_both["width"]
        ax.spines['top'].set_linewidth(outline_width)
        ax.spines['right'].set_linewidth(outline_width)
        ax.spines['bottom'].set_linewidth(outline_width)
        ax.spines['left'].set_linewidth(outline_width)
        
        if add_coef:
            # Increase the height of the whitespace in the plot
            whitespace_mult = add_coef_text_kwargs.pop("whitespace_pad")
            max_obs = np.max(column_totals)
            max_y = max_obs*whitespace_mult
            ax.set_ylim([0,max_y])
            
            # Get distance to add to each text y position
            labelpad_y = add_coef_text_kwargs.pop("labelpad_y")
            # Get distance for each text x position depending on the length
            # of the string. Because its a function of the length, this helps
            # to center the string perfectly.
            labelpad_x = add_coef_text_kwargs.pop("labelpad_x")
            for i,value in enumerate(regressor.coef_.ravel()[column_idx]):
                print_string = "{:.2f}".format(value)
                pos_x = x[i] + len(print_string)*labelpad_x
                pos_y = column_totals[i] + labelpad_y
                ax.text(pos_x,pos_y,print_string,
                        **add_coef_text_kwargs)
        
        # Now add coef if true
        if add_obs:
            # Increase the height of the whitespace in the plot
            whitespace_mult = add_obs_text_kwargs.pop("whitespace_pad")
            max_obs = np.max(column_totals)
            max_y = max_obs*whitespace_mult
            ax.set_ylim([0,max_y])
            
            # Get distance to add to each text y position
            labelpad_y = add_obs_text_kwargs.pop("labelpad_y")
            # Get distance for each text x position depending on the length
            # of the string. Because its a function of the length, this helps
            # to center the string perfectly.
            labelpad_x = add_obs_text_kwargs.pop("labelpad_x")
            for i,value in enumerate(column_totals):
                print_string = "{}".format(int(value))
                pos_x = x[i] + len(print_string)*labelpad_x
                pos_y = value + labelpad_y
                ax.text(pos_x,pos_y,print_string,
                        **add_obs_text_kwargs)
                
                
        plt.tight_layout()
        if len(figname) > 0:
            fig.savefig(figname)
        
    
    def similarity_matrix(self,neighborhood_df,metric="TC",
                          exclude_add_prop=True,
                          low_memory=False):
        """
        Constructs similarity matrix using the input neighborhood_df and 
        the input metric. 
        
        neighborhood_df: pandas.DataFrame
            Dataframe created by the constructor class. 
        metric: 
            Can be any scipy pdist metric or TC. 
        low_memory: bool
            Recommended to be true if using for a large number of molecules 
            (> 1000). If True, uses a low memory method should be used in the 
            computation of the TC metric. 
            
        """
        if "TC" not in implemented_metrics:   
            implemented_metrics.append("TC")
        if metric not in implemented_metrics:
            raise Exception("User used metric {} for similarity matrix. "
                    .format(metric) +
                    "Please use on of the implemented metrics: {}."
                    .format(implemented_metrics))
            
        if exclude_add_prop:
            column_idx = np.arange(len(self.add_prop), 
                                   len(neighborhood_df.columns),
                                   1)
        else:
            column_idx = np.arange(0,
                                   len(neighborhood_df.columns),
                                   1)
            
        values = neighborhood_df.values[:,column_idx]
            
        # Using any metric which is implemented for scipy pdist
        if metric != "TC":
            diff_matrix = squareform(pdist(values, metric=metric))
        
        # Using Tanimato Coefficient to compute similarity
        else: 
            diff_matrix = np.zeros((len(neighborhood_df.index),
                                    len(neighborhood_df.index)))
            if not low_memory:
                # First find NxN pairwise matrix of the number of fragments for 
                # each pair of molecules
                total_fragments = np.sum(values, axis=1)
                total_fragments_matrix = total_fragments + total_fragments[:,None]
                
                # Find where both have a nonzero value for the fragment
                nonzero_vectors = np.logical_and(values,values[:,None])
                # Find which has the minimum value for nonzero because that's  
                # the maximum number which can actually match 
                max_of_fragments = np.minimum(values,values[:,None])
                                
                # Mask maximum fragments by the nonzero mask
                c_array = np.ma.masked_array(max_of_fragments, 
                                             mask=nonzero_vectors, 
                                             fill_value=0).data
                
                # Now some the matching fragments for each pair of structure
                c_array = np.sum(c_array,axis=-1)
                
                # Compute the TC matrix as the diff_df
                diff_matrix = c_array / (total_fragments_matrix - c_array)
            
            # Low memory version follows the same steps as the above code
            # but loops over rows of the value matrix instead of computing
            # all similarities in memory at the same time.
            else:
                
                for i,row in enumerate(values):
                    print(i)
                    row = row[None,:]
                    total_fragments = np.sum(row)
                    # Get pairwise number of total fragments between the row
                    # and the rest of the molecules
                    total_fragments_matrix = total_fragments + np.sum(values,-1)
                    
                    # Find where the row and the other molecules have matching
                    # nonzero fragment values
                    nonzero_vectors = np.logical_and(row,values)
                                        
                    max_of_fragments = np.minimum(row,values)
                    c_array = np.ma.masked_array(max_of_fragments, 
                                             mask=nonzero_vectors, 
                                             fill_value=0).data
                                                 
                    c_array = np.sum(c_array,axis=-1)
                    if np.sum(c_array) == 0:
                        diff_matrix[i,:] = np.zeros(values.shape[0])
                    else:
                        diff_matrix[i,:] = c_array / (total_fragments_matrix - 
                                                      c_array)
                
        diff_df = pd.DataFrame(data=diff_matrix,
                               columns=neighborhood_df.index,
                               index=neighborhood_df.index)
        
        return diff_df
    
    
    def plot_similarity(self, diff_df, figname='',figsize=(10,10)):
        """
        Plot a similarity matrix as a color matrix. 
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(diff_df.values,interpolation='nearest')
        fig.colorbar(im)
        
        if len(figname) > 0:
            fig.savefig(figname)
            
        
        

if __name__ == "__main__":
    pass 
    
