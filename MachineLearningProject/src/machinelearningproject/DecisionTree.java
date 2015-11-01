/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import java.io.Serializable;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
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
    
    public static void main (String[] args)
    { 
        try {
            DecisionTree id3 = new DecisionTree();
//            ConverterUtils.DataSource source = new ConverterUtils.DataSource("D:\\spambase.arff");
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("D:\\weather-nominal.arff");
            Instances instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes()-1);
            
            id3.mainTree = id3.mainTree.buildTree(instances);
            id3.mainTree.printTree(id3.mainTree);
            
//            Random rand_ = new Random(5000);  
//            instances.randomize(rand_);

            //Evaluation
            System.out.println("\n==== EVALUASI CLASSIFIER BUATAN ====");        
            Evaluation eval = new Evaluation(instances);

            System.out.println("\nCross-Validation 10 Folds Evaluation");
            Random rand = new Random(1);        
            eval.crossValidateModel(id3, instances, 10, rand); 
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception ex) {
            Logger.getLogger(DecisionTree.class.getName()).log(Level.SEVERE, null, ex);         
        } 
    }
}
