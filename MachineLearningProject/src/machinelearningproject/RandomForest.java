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
public class RandomForest extends AbstractClassifier implements Serializable{
    
    public ArrayList<Tree> dtrees;
    
    public RandomForest()
    {
        dtrees = new ArrayList();
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        RFTree rftree = new RFTree();
        dtrees = rftree.buildRandomForest(i, 5);
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
    
    public static void main (String[] args)
    { 
        try {
            RandomForest rf = new RandomForest();
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("D:\\spambase.arff");
//            ConverterUtils.DataSource source = new ConverterUtils.DataSource("D:\\weather-nominal.arff");
            Instances instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes()-1);
                        
//            Random rand_ = new Random(5000);  
//            instances.randomize(rand_);

            //Evaluation
            System.out.println("\n==== EVALUASI CLASSIFIER BUATAN ====");        
            Evaluation eval = new Evaluation(instances);

            System.out.println("\nCross-Validation 10 Folds Evaluation");
            Random rand = new Random(100);        
            eval.crossValidateModel(rf, instances, 10, rand); 
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception ex) {
            Logger.getLogger(DecisionTree.class.getName()).log(Level.SEVERE, null, ex);         
        } 
    }
}
