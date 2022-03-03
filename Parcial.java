package praktika;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import static weka.classifiers.lazy.IBk.TAGS_WEIGHTING;
import static weka.classifiers.lazy.IBk.WEIGHT_INVERSE;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;
import static weka.classifiers.lazy.IBk.WEIGHT_SIMILARITY;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Parcial {

	public static void main(String[] args) throws Exception {
		
		
		
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		
		System.out.println("Atributu kop: " + data.numAttributes());
		System.out.println("Instantzia totalak " + data.numInstances());
		System.out.println("Klaseak zenbat balio har ditzake: " + data.numClasses() +" balio har ditzake");
		System.out.println("Klaseak har ditzakeen balio ezberdinak " + data.numDistinctValues(data.classIndex()));
		System.out.println("Azken aurreko atributuak dituen missin value: " + data.attributeStats(data.numAttributes() - 2).missingCount);
		
		
		for (int i = 0; i < data.numClasses(); i++) {
			System.out.println("Klasearen balioak , " + i + ". balioa, hurrengoa da " 
		+ data.attribute(data.classIndex()).value(i));

		}
		
		//Klase minoritarioa atera
		
		int min = Integer.MAX_VALUE;
		int minClassIndex = 0;
		for (int i = 0; i < data.numClasses(); i++) {
			int x = data.attributeStats(data.classIndex()).nominalCounts[i];
			System.out.println(data.attribute(data.classIndex()).value(i) + "--> " + x + " instantzia kopurua");
			if(x < min){
				min=x;
				minClassIndex = i;
			}
		}
		
		
		System.out.println("Klase minoritarioa: " + data.attribute(data.classIndex()).value(minClassIndex));
		System.out.println();
		System.out.println("---------------kNN Parametro Ekorketa-----------");
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

	}
	
	
	private static void ebaluazio_ez_zintzoa(Instances data) throws Exception {
		System.out.println("Ebaluazio ez_zintzoa erabilita :");
		
		
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data);
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(model, data);
		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());
		System.out.println("Accuracy " + eval.pctCorrect());

	}
	
	private static void hold_out(Instances data) throws Exception {
		System.out.println("Ebaluazio hold out erabilita :");
		
		
		Randomize randomFilter = new Randomize();
		randomFilter.setInputFormat(data);
		randomFilter.setRandomSeed(1);
		Instances randomData = Filter.useFilter(data, randomFilter);
		
		
		RemovePercentage removeFilter = new RemovePercentage();
		removeFilter.setInputFormat(randomData);
		removeFilter.setPercentage(30);
		Instances train = Filter.useFilter(randomData, removeFilter);
		System.out.println("Test instantziak: " + train.numInstances());
		
		
		removeFilter.setInputFormat(randomData);
		removeFilter.setPercentage(30);
		removeFilter.setInvertSelection(true);
		Instances test = Filter.useFilter(randomData, removeFilter);
		System.out.println("Test instantziak: " + test.numInstances());
		
		train.setClassIndex(data.numAttributes() - 1);
		test.setClassIndex(data.numAttributes() - 1);
		
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(model, test);
		
		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());

		System.out.println("Accuracy " + eval.pctCorrect());
		
	}
	
	private static void k_fold_crossValidation(Instances data) throws Exception {
		System.out.println("Ebaluazio k-foldCrossValidation erabilita :");
		
		NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 5, new Random(1));
		
		
		System.out.println("Lortutako datuak :" + eval.toSummaryString());
		System.out.println("Nahasmen matrizea " + eval.toMatrixString());
		System.out.println(eval.toClassDetailsString());
		System.out.println("Accuracy: " + eval.pctCorrect());
		System.out.println("Recall: " + eval.recall(0));
		System.out.println("Weighted Recall: " + eval.weightedRecall());
		
	}
	
	private static void parametro_ekorketa(Instances data, int minClass, String direktorio) throws Exception {
		
		int kkopurua = data.numInstances()/2;
		double maxFMeasure = Double.MIN_VALUE;
		int kaux = 0;
		int daux = 0;
		int waux = 0;
		
		LinearNNSearch euclideanDistance = new LinearNNSearch();
		euclideanDistance.setDistanceFunction(new EuclideanDistance());
		LinearNNSearch manhattanDistance = new LinearNNSearch();
		manhattanDistance.setDistanceFunction(new ManhattanDistance());
		LinearNNSearch minkowskiDistance = new LinearNNSearch();
		minkowskiDistance.setDistanceFunction(new MinkowskiDistance());
		
		LinearNNSearch[] distantziak = new LinearNNSearch[]{euclideanDistance, manhattanDistance, minkowskiDistance};
		SelectedTag[] tags = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};
		
		
		for (int i = 0; i < kkopurua; i++) {
			IBk ibk = new IBk();
			ibk.setKNN(i);
			for (int d = 0; d < distantziak.length; d++) {
				ibk.setNearestNeighbourSearchAlgorithm(distantziak[d]);
			
			
				for (int w = 0; w < tags.length; w++) {
					ibk.setDistanceWeighting(tags[w]);
					
					//ibk.buildClassifier(data);
					
					Evaluation eval = new Evaluation(data);
					eval.crossValidateModel(ibk, data, 5, new Random(1));
					
					if(eval.fMeasure(minClass) > maxFMeasure){
						maxFMeasure = eval.fMeasure(minClass);
						
						kaux = i;
						daux = d;
						waux = w;
						
					}
				}
			}	
		}
		
		System.out.println("F-measure maximoa --> " + maxFMeasure);
		System.out.println("K hoberena --> " + kaux);
		System.out.println("Distantzia mota hoberena -->" + distantziak[daux].getDistanceFunction().getClass().toString());
		System.out.println("Tag -> " + tags[daux].getSelectedTag().getReadable());
		
		
		IBk ibk = new IBk();
		ibk.setKNN(kaux);
		ibk.setNearestNeighbourSearchAlgorithm(distantziak[daux]);
		ibk.setDistanceWeighting(tags[waux]);
		ibk.buildClassifier(data);
		
		SerializationHelper.write(direktorio, ibk);
			
		DataSource sourceTest = new DataSource("/path/test.arff");
		Instances test = sourceTest.getDataSet();
		test.setClassIndex(data.numAttributes() - 1);
		
		
		Classifier ibk2 = (Classifier) SerializationHelper.read(direktorio);
		Evaluation iragarpenEval = new Evaluation(test);
		iragarpenEval.evaluateModel(ibk2, test);
		
		int i =0;
		for (Prediction p : iragarpenEval.predictions()) {
			String iragarpena = data.attribute(data.classIndex()).value((int)p.predicted());
			String erreala = data.attribute(data.classIndex()).value((int)p.actual());
			String errorea = "";
			
			if(Double.isNaN(p.actual())){
				iragarpena = "?";
			}
			if(iragarpena!=erreala){
				errorea = "$";
			}else{
				errorea = "-";
			}
			
			System.out.println("\t"+i+"\t\t"+iragarpena+"\t\t"+erreala+"\t\t   "+errorea);
			i++;
		}
		
		
	}
	
	private static void SMO(Instances data) throws Exception{
		weka.classifiers.functions.SMO smo = new weka.classifiers.functions.SMO();
		
		PolyKernel kernel = new PolyKernel();
		
		double fmax = Double.MIN_VALUE;
		double exponent = 0.0;
		
		for (int i = 0; i < 6; i++) {
			kernel.setExponent((double)i);
			smo.setKernel(kernel);
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(smo, data, 3, new Random(i));
			
			if(fmax < eval.weightedFMeasure()){
				fmax = eval.weightedFMeasure();
				exponent = (double)i;
			}
			
		}
		
		kernel.setExponent(exponent);
		smo.setKernel(kernel);
		smo.buildClassifier(data);
	}
	
	
	private static void fitxategia_idatzi(Evaluation ev, String direktorio)  {
		try {
			
			FileWriter file = new FileWriter(direktorio);
			
			PrintWriter pw = new PrintWriter(file);
			
			pw.println("Direktorioa -->" + direktorio);

			pw.println("Nahasmen matrizea -->" + ev.toMatrixString());
			
			pw.close();
		} catch (Exception e) {
			
			e.printStackTrace();
		}
	}
	
	
	

}
