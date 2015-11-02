/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Raches
 */
public class RandomForest extends AbstractClassifier implements Serializable{
    
    public ArrayList<Tree> dtrees;
    public int numTrees;
    
    public RandomForest(int numTrees)
    {
        this.dtrees = new ArrayList();
        this.numTrees = numTrees;
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        RFTree rftree = new RFTree();
        dtrees = rftree.buildRandomForest(i, numTrees);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        HashMap<String, Integer> classMap = new HashMap<>();
        
        for (int i = 0; i < dtrees.size(); i++){
            String key = dtrees.get(i).traverseTree(instance);
            if (classMap.isEmpty() || !classMap.containsKey(key)) {
                classMap.put(key, 1);
            } else {
                if (classMap.containsKey(key)){
                    classMap.put(key, classMap.get(key) + 1);
                }
            }
        }
        Iterator<String> keySetIterator = classMap.keySet().iterator();
        String modeClass = "";
        int count = 0;
        while(keySetIterator.hasNext()){ 
            String key = keySetIterator.next();
            if (count < classMap.get(key)){
                modeClass = key;
                count = classMap.get(key);
            }
        }
        
        double result = 0.0;
        for(int i = 0; i < instance.numClasses(); i++){
            if(modeClass.equals(instance.attribute(instance.classIndex()).value(i))){
                result = (double) i;
            }
        }
        
        return result;
    }
}
