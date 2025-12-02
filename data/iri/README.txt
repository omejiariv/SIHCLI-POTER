README

This folder stores the archive of our ENSO forecasts from 2002 to the present.

SST anomalies are speficied in tenths of a degree for a specific season, with a season being the
average of three months.

* ensofcst_ALL
  This is the raw data of our forecast in a Fortran formatted file, without the NOAA CPC consolidated
  forecast included. 

* ensofcst_cpc_ALL
  This is the raw data of our forecast in a Fortran formatted file with only the forecasts that include
  the CPC consolidated forecast

* enso_plumes.json
  This is a JSON formatted file of
  { years: [
      {
        year: year,
        months: [
          {
	    month: 0..11,
	    models: [{model}, {model}, ...],
	    observed: [
	      {
	        data: anomaly,
		month: season observed anomaly
	      },
	      {
	        data: anomaly,
		month: month observed data
	      }
	  },...
        ]
      }, ...
    ]
   }
   where a model contains:
       data: [float, float, float, float, float, float, float, float, float], # 9 seasons of anomalies
       model: modelname
       type: Dynamical | Statistical | CPC

  The climatological period for model the data is variable. We get our values from our partners' models.
  It is possible at times that differnt models during the same month are using different climatological
  periods for their anomaly values.

* enso_cpc_prob.json
  NOAA CPC predictions for the upcoming 9 seasons, which include the previous month.
  The official CPC ENSO probability forecast, based on a consensus of CPC and IRI forecasters. It is updated
  during the first half of the month, in association with the official CPC ENSO Diagnostic Discussion. It is
  based on observational and predictive information from early in the month and from the previous month. It uses
  human judgment in addition to model output, updated on the 2nd thursday of every month.

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