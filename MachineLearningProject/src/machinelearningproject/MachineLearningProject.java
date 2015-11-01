/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearningproject;
import static java.lang.Math.round;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Raches
 */
public class MachineLearningProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        DataSource source = new DataSource("D:\\spambase.arff");
//        DataSource source = new DataSource("D:\\weather-nominal.arff");
        Instances instances = source.getDataSet();
        int numAttr = instances.numAttributes();
        int k = (int)round(sqrt(numAttr));
        System.out.println("K Fraction "+k);
        instances.setClassIndex(instances.numAttributes() -1);
        
//        System.out.println(instances);
//        System.out.println(instances.attribute(instances.numAttributes()-1).name());
//        System.out.println("num of instances " + instances.size());
        
        DecisionTree tr = new DecisionTree();
//        System.out.println(tr.calculateClassEntropy(instances));
//        tr.calculateEntropy(instances, instances.numAttributes()-1);
//        for (int i=0;i<instances.numAttributes()-1;i++){
//            System.out.println(instances.attribute(i).name() +"-"+ tr.calculateInformationGain(instances, i, instances.numAttributes()-1));
//        }
//        System.out.println(instances.instance(1).value(3));
//        System.out.println(instances.get(1).stringValue(3);
        ArrayList<String> attrValues = new ArrayList();
        for (int i = 0; i < instances.numInstances(); i++){
            Instance instance = instances.get(i);
            String attrValue = instance.stringValue(0);
            if (attrValues.isEmpty() || !attrValues.contains(attrValue)){
                attrValues.add(attrValue);
            }
        }
    }
    
}
