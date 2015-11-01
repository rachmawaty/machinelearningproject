/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningproject;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class FileLoader {
    
    Instances Data;
    boolean debug;

    public void readwithoutmissing(String filepath) throws IOException, Exception{
        Instances data;
        if (filepath.endsWith(".arff")){
            ArffLoader loader = new ArffLoader();
            if (filepath.startsWith("http:") || filepath.startsWith("ftp:")){
                loader.setURL(filepath);
            } else {
                loader.setSource(new File(filepath));
            }
            data = loader.getDataSet();
            for(int i = 0; i < data.numAttributes();i++){
                data.deleteWithMissing(i);
            }
        } else if (filepath.endsWith(".csv")){
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filepath));
            data = loader.getDataSet();
            for(int i = 0; i < data.numAttributes();i++){
                data.deleteWithMissing(i);
            }
        } else {
            data = DataSource.read(filepath);
            for(int i = 0; i < data.numAttributes();i++){
                data.deleteWithMissing(i);
            }
        }
        Data = new Instances(data);
    }    
    
    public void readwithmissing(String filepath) throws IOException, Exception{
        Instances data;
        if (filepath.endsWith(".arff")){
            ArffLoader loader = new ArffLoader();
            if (filepath.startsWith("http:") || filepath.startsWith("ftp:")){
                loader.setURL(filepath);
            } else {
                loader.setSource(new File(filepath));
            }
            data = loader.getDataSet();
        } else if (filepath.endsWith(".csv")){
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filepath));
            data = loader.getDataSet();
        } else {
            data = DataSource.read(filepath);
        }
        Data = new Instances(data);
    }    
}
