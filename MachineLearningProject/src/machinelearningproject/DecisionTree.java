/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import java.io.Serializable;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author Raches
 */
public class DecisionTree extends AbstractClassifier implements Serializable{
       
    public Tree mainTree;
    
    public DecisionTree()
    {
        mainTree = new Tree();
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        mainTree = mainTree.buildTree(i);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        String classification = mainTree.traverseTree(instance);
        double result = 0.0;
        for(int i = 0; i < instance.numClasses(); i++){
            if(classification.equals(instance.attribute(instance.classIndex()).value(i))){
                result = (double) i;
            }
        }
        return result;
    }
}
