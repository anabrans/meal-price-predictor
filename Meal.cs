using Microsoft.ML.Runtime.Api;

namespace meal_price_predictor
{
    public class Meal
    {
        [Column("0")]
        public float Location;
        // HungerLevel is represented by 0 = starving, 1 = hungry, 2 = content, 3 = not hungry
        [Column("1")]
        public float HungerLevel;
        // HourofDay is represented between 0-23
        [Column("2")]
        public float HourofDay;
        // HappinessLevel is represented by 0 = unhappy, 1 = normal, 2 = happy
        [Column("3")]
        public float HappinessLevel;
        //RestaurantQualityLevel is represented on a scale of 1-5, 1 being the worst and 5 being the best
        [Column("4")]
        public float RestaurantQualityLevel;
        [Column("5")]
        public float MealCost;
    }
}