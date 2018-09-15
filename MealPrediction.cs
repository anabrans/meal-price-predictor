using Microsoft.ML.Runtime.Api;

namespace meal_price_predictor
{
    public class MealPrediction
    {
        [ColumnName("Score")]
        public float MealCost;
    }
}