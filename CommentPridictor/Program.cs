 // Ensure that this namespace is correct and matches your project structure.
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

class Program
{
    static void Main(string[] args)
    {
        string yelp_labelled = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        MLContext mlContext = new MLContext();

        // Load the data and split it
        TrainTestData splitDataView = LoadData(mlContext, yelp_labelled);

        // Build and train the model
        ITransformer model = BuildAndTrain(mlContext, splitDataView.TrainSet);

        // Get prediction for a sample review content
        GetPredictionForReviewContent(mlContext, model, "Fuck");

        Console.ReadLine();
    }

    static TrainTestData LoadData(MLContext mlContext, string dataPath)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, separatorChar: '\t', hasHeader: true);
        TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        return splitDataView;
    }

    static ITransformer BuildAndTrain(MLContext mlContext, IDataView splitTrainSet)
    {
        var estimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        var model = estimator.Fit(splitTrainSet);
        return model;
    }

    static void GetPredictionForReviewContent(MLContext mlContext, ITransformer model, string reviewContent)
    {
        PredictionEngine<SentimentData, SentimentPrediction> engine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var sampleStatement = new SentimentData
        {
            SentimentText = reviewContent
        };

        var predictionResult = engine.Predict(sampleStatement);

        Console.WriteLine($"Prediction: {(predictionResult.PredictedLabel ? "Positive" : "Negative")} Probability: {predictionResult.Probability}");
    }
}

// Assuming you have the following classes defined somewhere in your project
public class SentimentData
{
    // Assuming the first column in your data file is the text and the second column is the label
    [LoadColumn(0)]
    public string SentimentText { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; } // True for positive, false for negative
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }
}