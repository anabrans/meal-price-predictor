using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

using System.Collections.Generic;
using System.Linq;

using System.Diagnostics;

namespace meal_price_predictor
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "datasets", "meal-cost-train.csv");
        private static string TestDataPath => Path.Combine(AppPath, "datasets", "meal-cost-test.csv");
        private static string ModelPath => Path.Combine(AppPath, "MealCostModel.zip");

        private static async Task Main(string[] args) //If args[0] == "svg" a vector-based chart will be created instead a .png chart
        {
            // STEP 1: Create a model
            var model = await TrainAsync();
            var myMeal = new Meal(){
                HourofDay = 19,
                HungerLevel = 1,
                HappinessLevel = 1,
                RestaurantQualityLevel = 4,
                Location = 1,
                MealCost = 9.83f

            };
            Console.WriteLine("Hi");
            Console.WriteLine(myMeal.MealCost.ToString());
            // STEP2: Test accuracy
            Evaluate(model);

            // STEP 3: Make a test prediction
            var prediction = model.Predict(myMeal);
            Console.WriteLine(prediction);
            Console.WriteLine($"Predicted MealCost: {prediction.MealCost:0.####}, actual MealCost: 9.83");

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();
        }

        private static async Task<PredictionModel<Meal, MealPrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.
            var pipeline = new LearningPipeline
            {
                // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
                // all the column names and their types.
                new TextLoader(TrainDataPath).CreateFrom<Meal>(separator:','),
                
                // Transforms
                // When ML model starts training, it looks for two columns: Label and Features.
                // Label:   values that should be predicted. If you have a field named Label in your data type,
                //              no extra actions required.
                //          If you don't have it, like in this example, copy the column you want to predict with
                //              ColumnCopier transform:
                new ColumnCopier(("MealCost", "Label")),
                
                // CategoricalOneHotVectorizer transforms categorical (string) values into 0/1 vectors
             
                // Features: all data used for prediction. At the end of all transforms you need to concatenate
                //              all columns except the one you want to predict into Features column with
                //              ColumnConcatenator transform:
                new ColumnConcatenator("Features",
                "Location",
                    "HungerLevel",
                    "HourofDay",
                    "HappinessLevel",
                    "RestaurantQualityLevel",
                    "MealCost"),
                //FastTreeRegressor is an algorithm that will be used to train the model.
                new FastTreeRegressor()
            };

            Console.WriteLine("=============== Training model ===============");
            // The pipeline is trained on the dataset that has been loaded and transformed.
            var model = pipeline.Train<Meal, MealPrediction>();

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<Meal, MealPrediction> model)
        {
            // To evaluate how good the model predicts values, it is run against new set
            // of data (test data) that was not involved in training.
            var testData = new TextLoader(TestDataPath).CreateFrom<Meal>(separator: ',');

            // RegressionEvaluator calculates the differences (in various metrics) between predicted and actual
            // values in the test dataset.
            var evaluator = new RegressionEvaluator();

            Console.WriteLine("=============== Evaluating model ===============");

            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}, ideally should be around 2.8, can be improved with larger dataset");
            Console.WriteLine($"RSquared = {metrics.RSquared}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }

    }
}