# EDA findings (20 questions)

**Q1.** Prices are roughly uniform between ₹10L and ₹500L (mean 255L, median 254L, skew 0.01). There is no long right tail as in real markets.

**Q2.** Sizes are roughly uniform from 500 to 5000 sq ft (median 2747).

**Q3.** Median price per sq ft by type: {'Apartment': 9245.0, 'Independent House': 9305.0, 'Villa': 9190.0}. Differences are small, so type is a weak price driver here.

**Q4.** Correlation between size and price is -0.003, i.e. effectively none. Larger homes are NOT priced higher in this dataset; instead price per sq ft falls as size rises (corr -0.61), which is a mathematical by-product of price being independent of size.

**Q5.** 7.9% of properties are IQR-outliers on price per sq ft (high side; 0.0% on the low side), driven by small homes with high prices; max ₹99,182/sq ft. Size has no outliers (bounded 500-5000). Outliers were kept for modelling because they are valid small-size/high-price combinations, and the tree models are robust to them.

**Q6.** Highest: Karnataka (₹13,252); lowest: Punjab (₹12,931). Spread between states is only 2%.

**Q7.** Highest: Bangalore (258L); lowest: Cuttack (251L). Cities are all within 251-258L, i.e. very flat.

**Q8.** Median age per locality ranges 16-21 years (overall median 18). Oldest: Locality_403, Locality_72, Locality_1; newest: Locality_443, Locality_10, Locality_118. Caveat: every locality id appears in 42 cities on median, so locality ids are not real neighbourhoods and these differences are sampling noise.

**Q9.** BHK is near-uniform in every city (each of 1-5 BHK is roughly 20%; max city deviation 2.0%).

**Q10.** Top-5 localities by median ₹/sq ft (min 200 listings): Locality_146, Locality_207, Locality_164, Locality_44, Locality_360. The data has no sale-date field, so 'trend' is shown by year-built cohort; lines are noisy with no consistent upward trend, so no appreciation signal can be read from the data itself (this is why the 5-year price is an assumption-based projection, not a learned one). Locality ids are not nested in cities, so this ranking is noise.

**Q11.** Only mechanical relationships appear: Size_in_SqFt vs PPSF (-0.61); Price_in_Lakhs vs PPSF (0.56); Infra_Score vs Nearby_Schools (0.54). Largest |r| among raw, non-derived attributes (BHK, size, price, age, schools, hospitals, floors): 0.006.

**Q12.** Correlation between schools count and price per sq ft is -0.000: no relationship. Locations with more schools are not priced higher.

**Q13.** Correlation between hospitals count and price per sq ft is -0.000: no relationship. Locations with more hospitals are not priced higher.

**Q14.** Mean price (lakhs): {'Furnished': 254.4, 'Semi-furnished': 254.3, 'Unfurnished': 255.0}. Differences under 1%, so furnishing does not drive price.

**Q15.** Mean ₹/sq ft by facing: {'East': 13024.0, 'North': 13025.0, 'South': 13044.0, 'West': 13140.0}. Effectively identical.

**Q16.** Owner types are evenly split: {'Broker': 83479, 'Owner': 83268, 'Builder': 83253}.

**Q17.** {'Under_Construction': 125035, 'Ready_to_Move': 124965}. Only these two statuses exist in the data (no 'Sold').

**Q18.** Mean price with/without parking (lakhs): {0: 254.4, 1: 254.7}. No meaningful effect.

**Q19.** Listings with each amenity average ₹13,049-13,097/sq ft (spread 0.4%). Price per sq ft by number of amenities ranges only 13,015-13,102; amenities do not drive price.

**Q20.** ₹/sq ft by transport level: {'High': 13027.0, 'Low': 13086.0, 'Medium': 13062.0} (flat). Good-investment rate: {'High': 0.354, 'Low': 0.196, 'Medium': 0.348}. The rate rises with transport access because the investment rule awards points for it (rule effect, not market evidence).

**Extra (brief step 2).** The dataset has no crime-rate column, so that analysis cannot be done. Infrastructure score vs assumption-based projected value is shown in `extra_infra_vs_future.png`; it is mildly positive only because the growth-rate assumption rewards infrastructure.
