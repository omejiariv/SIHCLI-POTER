* enso_iri_prob.json
  The IRI predictions for the upcoming 9 seasons, starting at the current month.
  This is a purely objective ENSO probability forecast, based on regression, using as input the model predictions
  from the plume of dynamical and statistical forecasts shown in Fig. 4. Each of the forecasts is weighted equally.
  It is updated near or just after the middle of the month, using forecasts from the plume models that are run in
  the first half of the month. It does not use any human interpretation or judgment. It is updated on or about the
  19th of every month.

  This is a JSON formatted file in the format:
  { years: [
    {
      year: year,
      months: [
        {
          month: 0..11,
          probabilities: [
            {
              elnino:
              lanina:
              neutral:
              season: season
             }, ...
          ]
        }]
    }]
  }
