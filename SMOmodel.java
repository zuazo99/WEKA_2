package SMO;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class SMOmodel {

	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		
		
		Randomize filterRandom = new Randomize();
		filterRandom.setRandomSeed(1); //esto se usa para el for aqui se coloca la i sino hay for se pone 1
		filterRandom.setInputFormat(data); //siempre que modifiques algo le recuerdas al filtro su formato
		Instances RandomData = Filter.useFilter(data, filterRandom);

		RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(RandomData); //Preparas el filtro.
        filterRemove.setPercentage(30); //Ajustas la cantidad de datos que quieres borrar --> En este caso --> 30% borras y te quedas 70%    
        Instances train = Filter.useFilter(RandomData,filterRemove);
        
        
        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30);	
        filterRemove.setInvertSelection(true);
        Instances test = Filter.useFilter(RandomData,filterRemove);
        
        
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1 );
		
		
		
		//1.1 Datuen analisia
		
		System.out.println("Atributu kop : " + data.numAttributes());
		System.out.println("Instantzia kop : " + data.numInstances());
		
		System.out.println("Klasea atributua har ditzakeen balio ezberdinak :" + data.numDistinctValues(data.classIndex()));
		
		for (int i = 0; i < data.numClasses(); i++) {
			
			int maiztasuna = data.attributeStats(data.classIndex()).nominalCounts[i];
			String atributua = data.attribute(data.classIndex()).value(i);
			System.out.println(atributua + " balioa hurrengo agerpen kopurua " + maiztasuna);
			
		}
		
		//Klase minoritarioa ateratzeko
		
		int min = Integer.MAX_VALUE;
		int minClassValue = 0;
		
		for (int i = 0; i < data.numClasses(); i++) {
			int x = data.attributeStats(data.classIndex()).nominalCounts[i];
			System.out.println(data.attribute(data.classIndex()).value(i) + " klase minoritarioa");
			if(x < min){
				min = x;
				minClassValue = i;
			}
		}
		System.out.println("Kaixo");
		//1.2 Klasifikatzailea --> SMO
		
		SMO smo = new SMO();
		//smo.buildClassifier(data);
		PolyKernel kernel = new PolyKernel();
		
		double max = 0.0;
		double exponent = 0.0;
		
		for (int i = 1; i < 2 ; i++) {
			
			kernel.setExponent((double)i);
			smo.setKernel(kernel);
			Evaluation eval = new Evaluation(data);
			
			eval.crossValidateModel(smo, train, 2, new Random(1));
			 
			 if(max < eval.weightedFMeasure() ){
				 max = eval.weightedFMeasure();
				 exponent = (double)i;
			 }
			 System.out.println("Berretzaileen balioa: " + (double)i + " neurria --> " + eval.weightedFMeasure());
		}
		
		kernel.setExponent(exponent);
		smo.setKernel(kernel);
		smo.buildClassifier(data);
		
		SerializationHelper.write(args[2], smo);
		
		//1.3 Orain gorde behar dugu lortu dugun modeloa
		
		Evaluation ezZintzoa = new Evaluation(data);
		ezZintzoa.evaluateModel(smo, data);
		
		try{
		FileWriter file = new FileWriter(args[3]);
		PrintWriter pw = new PrintWriter(file);
		
		pw.println("Precision balioa klase minoritariokoa: " + ezZintzoa.precision(minClassValue));
		pw.println("Recall balioa klase minoritariokoa: " + ezZintzoa.recall(minClassValue));
		pw.println("F-measure balioa klase minoritariokoa: " + ezZintzoa.fMeasure(minClassValue));
		pw.println(ezZintzoa.toMatrixString());
		
		
		
		
		
//		DataSource sourceTest = new DataSource(args[1]);
//		Instances test = sourceTest.getDataSet();
//		data.setClassIndex(data.numAttributes() - 1);
		
		
		//1.4 IRAGARPENAK
		
		Classifier rSMO = (Classifier) SerializationHelper.read(args[2]); 
		Evaluation iragarpenEval = new Evaluation(data);
		iragarpenEval.evaluateModel(rSMO, test);
		
		
		
		FileWriter filePre = new FileWriter(args[4]);
		PrintWriter pwPre = new PrintWriter(filePre);
			
		int i = 0;
		for (Prediction p : iragarpenEval.predictions()) {
			String iragarpena = data.attribute(data.classIndex()).value((int) p.predicted());
			String erreala = data.attribute(data.classIndex()).value((int)p.actual());
			
			String errorea="";
			if(Double.isNaN(p.actual())) {
				iragarpena="?";
			}
			if(iragarpena!=erreala) {
				errorea="$";
			}
			else {
				errorea="-";
			}
			
			System.out.println("\t"+i+"\t\t"+iragarpena+"\t\t"+erreala+"\t\t   "+errorea);
			pwPre.println("\t"+i+"\t\t"+iragarpena+"\t\t"+erreala+"\t\t   "+errorea);
			i ++;
			
		}
			
			
			
		}catch (Exception e) {
				// TODO: handle exception
			}
	
		
		
		
		
			
		
		

	}
	
	private static void holdOut(Instances data, int i) throws Exception{
		
		Randomize filterRandom = new Randomize();
		filterRandom.setRandomSeed(i); //esto se usa para el for aqui se coloca la i sino hay for se pone 1
		filterRandom.setInputFormat(data); //siempre que modifiques algo le recuerdas al filtro su formato
		Instances RandomData = Filter.useFilter(data, filterRandom);

		RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(RandomData); //Preparas el filtro.
        filterRemove.setPercentage(30); //Ajustas la cantidad de datos que quieres borrar --> En este caso --> 30% borras y te quedas 70%    
        Instances train = Filter.useFilter(RandomData,filterRemove);
        
        
        filterRemove.setInputFormat(RandomData);
        filterRemove.setPercentage(30);	
        filterRemove.setInvertSelection(true);
        Instances test = Filter.useFilter(RandomData,filterRemove);
        
        
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1 );
        
        
        NaiveBayes model = new NaiveBayes();
        //SMO mirar
        //SMO model1 = new SMO();
        model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        
     
//        eval.evaluateModel(model, test);
//        System.out.println(eval.toMatrixString());
//        System.out.println(eval.pctCorrect());
	}

}
