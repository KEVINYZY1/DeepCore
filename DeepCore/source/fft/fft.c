#include"../../include/fft/fft.h"

typedef struct fft_kernel_prop{
	char*        names[3];
	unsigned int argmask;
} fft_kernel_prop_t;

static const fft_kernel_prop_t fftkp_r2c[]=
{
	{ { "d_sfft16x16_r2c"       , "d_xfft16x16_r2c"       , "d_hfft16x16_r2c"        }, AM_3P_1S },
	{ { "d_sfft16x16_r2c_ext"   , "d_xfft16x16_r2c_ext"   , "d_hfft16x16_r2c_ext"    }, AM_3P_3S },
	{ { "d_sfft16x16_r2c_pad"   , "d_xfft16x16_r2c_pad"   , "d_hfft16x16_r2c_pad"    }, AM_3P_6S },
	{ { "d_sfft16x16_r2c_flip"  , "d_xfft16x16_r2c_flip"  , "d_hfft16x16_r2c_flip"	 }, AM_3P_3S },
	{ { "d_sfft32x32_r2c"       , "d_xfft32x32_r2c"       , "d_hfft32x32_r2c"        }, AM_3P_1S },
	{ { "d_sfft32x32_r2c_ext"   , "d_xfft32x32_r2c_ext"   , "d_hfft32x32_r2c_ext"    }, AM_3P_3S },
	{ { "d_sfft32x32_r2c_pad"   , "d_xfft32x32_r2c_pad"   , "d_hfft32x32_r2c_pad"    }, AM_3P_6S },
	{ { "d_sfft32x32_r2c_flip"  , "d_xfft32x32_r2c_flip"  , "d_hfft32x32_r2c_flip"	 }, AM_3P_3S },
	{ { "d_sfft64x64_r2c"       , "d_xfft64x64_r2c"       , "d_hfft64x64_r2c"        }, AM_3P_1S },
	{ { "d_sfft64x64_r2c_ext"   , "d_xfft64x64_r2c_ext"   , "d_hfft64x64_r2c_ext"    }, AM_3P_3S },
	{ { "d_sfft64x64_r2c_pad"   , "d_xfft64x64_r2c_pad"   , "d_hfft64x64_r2c_pad"    }, AM_3P_6S },
	{ { "d_sfft64x64_r2c_flip"  , "d_xfft64x64_r2c_flip"  , "d_hfft64x64_r2c_flip"	 }, AM_3P_3S },
	{ { "d_sfft128x128_r2c"     , "d_xfft128x128_r2c"     , "d_hfft128x128_r2c"      }, AM_3P_1S },
	{ { "d_sfft128x128_r2c_ext" , "d_xfft128x128_r2c_ext" , "d_hfft128x128_r2c_ext"  }, AM_3P_3S },
	{ { "d_sfft128x128_r2c_pad" , "d_xfft128x128_r2c_pad" , "d_hfft128x128_r2c_pad"  }, AM_3P_6S },
	{ { "d_sfft128x128_r2c_flip", "d_xfft128x128_r2c_flip", "d_hfft128x128_r2c_flip" }, AM_3P_3S }
};

static const fft_kernel_prop_t fftkp_c2r[]=
{
	{ { "d_sfft16x16_c2r"            , "d_xfft16x16_c2r"            , "d_hfft16x16_c2r"             }, AM_3P_3S },
	{ { "d_sfft16x16_c2r_frelu"      , "d_xfft16x16_c2r_frelu"      , "d_hfft16x16_c2r_frelu"       }, AM_3P_4S },
	{ { "d_sfft16x16_c2r_felu"       , "d_xfft16x16_c2r_felu"       , "d_hfft16x16_c2r_felu"        }, AM_3P_4S },
	{ { "d_sfft16x16_c2r_bias"       , "d_xfft16x16_c2r_bias"       , "d_hfft16x16_c2r_bias"        }, AM_4P_3S },
	{ { "d_sfft16x16_c2r_bias_relu"  , "d_xfft16x16_c2r_bias_relu"  , "d_hfft16x16_c2r_bias_relu"   }, AM_4P_4S },
	{ { "d_sfft16x16_c2r_bias_elu"   , "d_xfft16x16_c2r_bias_elu"   , "d_hfft16x16_c2r_bias_elu"    }, AM_4P_4S },
	{ { "d_sfft16x16_c2r_xdiff"      , "d_xfft16x16_c2r_xdiff"      , "d_hfft16x16_c2r_xdiff"       }, AM_4P_3S },
	{ { "d_sfft16x16_c2r_brelu"      , "d_xfft16x16_c2r_brelu"      , "d_hfft16x16_c2r_brelu"       }, AM_4P_4S },
	{ { "d_sfft16x16_c2r_belu"       , "d_xfft16x16_c2r_belu"       , "d_hfft16x16_c2r_belu"        }, AM_4P_4S },
	{ { "d_sfft16x16_c2r_filter"     , "d_xfft16x16_c2r_filter"     , "d_hfft16x16_c2r_filter"      }, AM_3P_3S },
	{ { "d_sfft32x32_c2r"            , "d_xfft32x32_c2r"            , "d_hfft32x32_c2r"             }, AM_3P_3S },
	{ { "d_sfft32x32_c2r_frelu"      , "d_xfft32x32_c2r_frelu"      , "d_hfft32x32_c2r_frelu"       }, AM_3P_4S },
	{ { "d_sfft32x32_c2r_felu"       , "d_xfft32x32_c2r_felu"       , "d_hfft32x32_c2r_felu"        }, AM_3P_4S },
	{ { "d_sfft32x32_c2r_bias"       , "d_xfft32x32_c2r_bias"       , "d_hfft32x32_c2r_bias"        }, AM_4P_3S },
	{ { "d_sfft32x32_c2r_bias_relu"  , "d_xfft32x32_c2r_bias_relu"  , "d_hfft32x32_c2r_bias_relu"   }, AM_4P_4S },
	{ { "d_sfft32x32_c2r_bias_elu"   , "d_xfft32x32_c2r_bias_elu"   , "d_hfft32x32_c2r_bias_elu"    }, AM_4P_4S },
	{ { "d_sfft32x32_c2r_xdiff"      , "d_xfft32x32_c2r_xdiff"      , "d_hfft32x32_c2r_xdiff"       }, AM_4P_3S },
	{ { "d_sfft32x32_c2r_brelu"      , "d_xfft32x32_c2r_brelu"      , "d_hfft32x32_c2r_brelu"       }, AM_4P_4S },
	{ { "d_sfft32x32_c2r_belu"       , "d_xfft32x32_c2r_belu"       , "d_hfft32x32_c2r_belu"        }, AM_4P_4S },
	{ { "d_sfft32x32_c2r_filter"     , "d_xfft32x32_c2r_filter"     , "d_hfft32x32_c2r_filter"      }, AM_3P_3S },
	{ { "d_sfft64x64_c2r"            , "d_xfft64x64_c2r"            , "d_hfft64x64_c2r"             }, AM_3P_3S },
	{ { "d_sfft64x64_c2r_frelu"      , "d_xfft64x64_c2r_frelu"      , "d_hfft64x64_c2r_frelu"       }, AM_3P_4S },
	{ { "d_sfft64x64_c2r_felu"       , "d_xfft64x64_c2r_felu"       , "d_hfft64x64_c2r_felu"        }, AM_3P_4S },
	{ { "d_sfft64x64_c2r_bias"       , "d_xfft64x64_c2r_bias"       , "d_hfft64x64_c2r_bias"        }, AM_4P_3S },
	{ { "d_sfft64x64_c2r_bias_relu"  , "d_xfft64x64_c2r_bias_relu"  , "d_hfft64x64_c2r_bias_relu"   }, AM_4P_4S },
	{ { "d_sfft64x64_c2r_bias_elu"   , "d_xfft64x64_c2r_bias_elu"   , "d_hfft64x64_c2r_bias_elu"    }, AM_4P_4S },
	{ { "d_sfft64x64_c2r_xdiff"      , "d_xfft64x64_c2r_xdiff"      , "d_hfft64x64_c2r_xdiff"       }, AM_4P_3S },
	{ { "d_sfft64x64_c2r_brelu"      , "d_xfft64x64_c2r_brelu"      , "d_hfft64x64_c2r_brelu"       }, AM_4P_4S },
	{ { "d_sfft64x64_c2r_belu"       , "d_xfft64x64_c2r_belu"       , "d_hfft64x64_c2r_belu"        }, AM_4P_4S },
	{ { "d_sfft64x64_c2r_filter"     , "d_xfft64x64_c2r_filter"     , "d_hfft64x64_c2r_filter"      }, AM_3P_3S },
	{ { "d_sfft128x128_c2r"          , "d_xfft128x128_c2r"          , "d_hfft128x128_c2r"           }, AM_3P_3S },
	{ { "d_sfft128x128_c2r_frelu"    , "d_xfft128x128_c2r_frelu"    , "d_hfft128x128_c2r_frelu"     }, AM_3P_4S },
	{ { "d_sfft128x128_c2r_felu"     , "d_xfft128x128_c2r_felu"     , "d_hfft128x128_c2r_felu"      }, AM_3P_4S },	
	{ { "d_sfft128x128_c2r_bias"     , "d_xfft128x128_c2r_bias"     , "d_hfft128x128_c2r_bias"      }, AM_4P_3S },
	{ { "d_sfft128x128_c2r_bias_relu", "d_xfft128x128_c2r_bias_relu", "d_hfft128x128_c2r_bias_relu" }, AM_4P_4S },
	{ { "d_sfft128x128_c2r_bias_elu" , "d_xfft128x128_c2r_bias_elu" , "d_hfft128x128_c2r_bias_elu"  }, AM_4P_4S },
	{ { "d_sfft128x128_c2r_xdiff"    , "d_xfft128x128_c2r_xdiff"    , "d_hfft128x128_c2r_xdiff"     }, AM_4P_3S },
	{ { "d_sfft128x128_c2r_brelu"    , "d_xfft128x128_c2r_brelu"    , "d_hfft128x128_c2r_brelu"     }, AM_4P_4S },
	{ { "d_sfft128x128_c2r_belu"     , "d_xfft128x128_c2r_belu"     , "d_hfft128x128_c2r_belu"      }, AM_4P_4S },
	{ { "d_sfft128x128_c2r_filter"   , "d_xfft128x128_c2r_filter"   , "d_hfft128x128_c2r_filter"    }, AM_3P_3S }
};

static const fft_kernel_prop_t cellfftkp_r2c[]=
{ 	
	{ { "d_sfft16x16_r2c_perm3d"     , "d_xfft16x16_r2c_perm3d"     , "d_hfft16x16_r2c_perm3d"      }, AM_3P_6S },
	{ { "d_sfft16x16_r2c_perm3d_ext" , "d_xfft16x16_r2c_perm3d_ext" , "d_hfft16x16_r2c_perm3d_ext"  }, AM_3P_6S },
	{ { "d_sfft16x16_r2c_perm3d_pad" , "d_xfft16x16_r2c_perm3d_pad" , "d_hfft16x16_r2c_perm3d_pad"  }, AM_3P_7S },
	{ { "d_sfft16x16_r2c_perm3d_flip", "d_xfft16x16_r2c_perm3d_flip", "d_hfft16x16_r2c_perm3d_flip" }, AM_3P_6S },
	{ { "d_sfft16x16_r2c_perm2d"     , "d_xfft16x16_r2c_perm2d"     , "d_hfft16x16_r2c_perm2d"      }, AM_3P_5S },
	{ { "d_sfft16x16_r2c_perm2d_ext" , "d_xfft16x16_r2c_perm2d_ext" , "d_hfft16x16_r2c_perm2d_ext"  }, AM_3P_5S },
	{ { "d_sfft16x16_r2c_perm2d_pad" , "d_xfft16x16_r2c_perm2d_pad" , "d_hfft16x16_r2c_perm2d_pad"  }, AM_3P_7S },
	{ { "d_sfft16x16_r2c_perm2d_flip", "d_xfft16x16_r2c_perm2d_flip", "d_hfft16x16_r2c_perm2d_flip" }, AM_3P_5S },
	{ { "d_sfft32x32_r2c_perm3d"     , "d_xfft32x32_r2c_perm3d"     , "d_hfft32x32_r2c_perm3d"      }, AM_3P_6S },
	{ { "d_sfft32x32_r2c_perm3d_ext" , "d_xfft32x32_r2c_perm3d_ext" , "d_hfft32x32_r2c_perm3d_ext"  }, AM_3P_6S },
	{ { "d_sfft32x32_r2c_perm3d_pad" , "d_xfft32x32_r2c_perm3d_pad" , "d_hfft32x32_r2c_perm3d_pad"  }, AM_3P_7S },
	{ { "d_sfft32x32_r2c_perm3d_flip", "d_xfft32x32_r2c_perm3d_flip", "d_hfft32x32_r2c_perm3d_flip" }, AM_3P_6S },
	{ { "d_sfft32x32_r2c_perm2d"     , "d_xfft32x32_r2c_perm2d"     , "d_hfft32x32_r2c_perm2d"      }, AM_3P_5S },
	{ { "d_sfft32x32_r2c_perm2d_ext" , "d_xfft32x32_r2c_perm2d_ext" , "d_hfft32x32_r2c_perm2d_ext"  }, AM_3P_5S },
	{ { "d_sfft32x32_r2c_perm2d_pad" , "d_xfft32x32_r2c_perm2d_pad" , "d_hfft32x32_r2c_perm2d_pad"  }, AM_3P_7S },
	{ { "d_sfft32x32_r2c_perm2d_flip", "d_xfft32x32_r2c_perm2d_flip", "d_hfft32x32_r2c_perm2d_flip" }, AM_3P_5S },
	{ { "d_sfft32x32_r2c_split"      , "d_xfft32x32_r2c_split"      , "d_hfft32x32_r2c_split"       }, AM_3P_9S },
	{ { "d_sfft32x32_r2c_split_pad"  , "d_xfft32x32_r2c_split_pad"  , "d_hfft32x32_r2c_split_pad"   }, AM_3P_9S }
};

static const fft_kernel_prop_t cellfftkp_c2r[]=
{ 	
	{{ "d_sfft16x16_c2r_perm3d"          , "d_xfft16x16_c2r_perm3d"          , "d_hfft16x16_c2r_perm3d"           }, AM_3P_5S },
	{{ "d_sfft16x16_c2r_perm3d_frelu"    , "d_xfft16x16_c2r_perm3d_frelu"    , "d_hfft16x16_c2r_perm3d_frelu"     }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_perm3d_felu"     , "d_xfft16x16_c2r_perm3d_felu"     , "d_hfft16x16_c2r_perm3d_felu"      }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_perm3d_bias"     , "d_xfft16x16_c2r_perm3d_bias"     , "d_hfft16x16_c2r_perm3d_bias"      }, AM_4P_5S },
	{{ "d_sfft16x16_c2r_perm3d_bias_relu", "d_xfft16x16_c2r_perm3d_bias_relu", "d_hfft16x16_c2r_perm3d_bias_relu" }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm3d_bias_elu" , "d_xfft16x16_c2r_perm3d_bias_elu" , "d_hfft16x16_c2r_perm3d_bias_elu"  }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm3d_xdiff"    , "d_xfft16x16_c2r_perm3d_xdiff"    , "d_hfft16x16_c2r_perm3d_xdiff"     }, AM_4P_5S },
	{{ "d_sfft16x16_c2r_perm3d_brelu"    , "d_xfft16x16_c2r_perm3d_brelu"    , "d_hfft16x16_c2r_perm3d_brelu"     }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm3d_belu"     , "d_xfft16x16_c2r_perm3d_belu"     , "d_hfft16x16_c2r_perm3d_belu"      }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm2d"          , "d_xfft16x16_c2r_perm2d"          , "d_hfft16x16_c2r_perm2d"           }, AM_3P_5S },
	{{ "d_sfft16x16_c2r_perm2d_frelu"    , "d_xfft16x16_c2r_perm2d_frelu"    , "d_hfft16x16_c2r_perm2d_frelu"     }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_perm2d_felu"     , "d_xfft16x16_c2r_perm2d_felu"     , "d_hfft16x16_c2r_perm2d_felu"      }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_perm2d_bias"     , "d_xfft16x16_c2r_perm2d_bias"     , "d_hfft16x16_c2r_perm2d_bias"      }, AM_4P_5S },
	{{ "d_sfft16x16_c2r_perm2d_bias_relu", "d_xfft16x16_c2r_perm2d_bias_relu", "d_hfft16x16_c2r_perm2d_bias_relu" }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm2d_bias_elu" , "d_xfft16x16_c2r_perm2d_bias_elu" , "d_hfft16x16_c2r_perm2d_bias_elu"  }, AM_4P_6S },
	{{ "d_sfft16x16_c2r_perm2d_xdiff"    , "d_xfft16x16_c2r_perm2d_xdiff"    , "d_hfft16x16_c2r_perm2d_xdiff"     }, AM_3P_5S },
	{{ "d_sfft16x16_c2r_perm2d_brelu"    , "d_xfft16x16_c2r_perm2d_brelu"    , "d_hfft16x16_c2r_perm2d_brelu"     }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_perm2d_belu"     , "d_xfft16x16_c2r_perm2d_belu"     , "d_hfft16x16_c2r_perm2d_belu"      }, AM_3P_6S },
	{{ "d_sfft16x16_c2r_filter"          , "d_xfft16x16_c2r_filter"          , "d_hfft16x16_c2r_filter"           }, AM_3P_5S },
	{{ "d_sfft32x32_c2r_perm3d"          , "d_xfft32x32_c2r_perm3d"          , "d_hfft32x32_c2r_perm3d"           }, AM_3P_5S },
	{{ "d_sfft32x32_c2r_perm3d_frelu"    , "d_xfft32x32_c2r_perm3d_frelu"    , "d_hfft32x32_c2r_perm3d_frelu"     }, AM_3P_6S },
	{{ "d_sfft32x32_c2r_perm3d_felu"     , "d_xfft32x32_c2r_perm3d_felu"     , "d_hfft32x32_c2r_perm3d_felu"      }, AM_3P_6S },
	{{ "d_sfft32x32_c2r_perm3d_bias"     , "d_xfft32x32_c2r_perm3d_bias"     , "d_hfft32x32_c2r_perm3d_bias"      }, AM_4P_5S },
	{{ "d_sfft32x32_c2r_perm3d_bias_relu", "d_xfft32x32_c2r_perm3d_bias_relu", "d_hfft32x32_c2r_perm3d_bias_relu" }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm3d_bias_elu" , "d_xfft32x32_c2r_perm3d_bias_elu" , "d_hfft32x32_c2r_perm3d_bias_elu"  }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm3d_xdiff"    , "d_xfft32x32_c2r_perm3d_xdiff"    , "d_hfft32x32_c2r_perm3d_xdiff"     }, AM_4P_5S },
	{{ "d_sfft32x32_c2r_perm3d_brelu"    , "d_xfft32x32_c2r_perm3d_brelu"    , "d_hfft32x32_c2r_perm3d_brelu"     }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm3d_belu"     , "d_xfft32x32_c2r_perm3d_belu"     , "d_hfft32x32_c2r_perm3d_belu"      }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm2d"          , "d_xfft32x32_c2r_perm2d"          , "d_hfft32x32_c2r_perm2d"           }, AM_3P_5S },
	{{ "d_sfft32x32_c2r_perm2d_frelu"    , "d_xfft32x32_c2r_perm2d_frelu"    , "d_hfft32x32_c2r_perm2d_frelu"     }, AM_3P_6S },
	{{ "d_sfft32x32_c2r_perm2d_felu"     , "d_xfft32x32_c2r_perm2d_felu"     , "d_hfft32x32_c2r_perm2d_felu"      }, AM_3P_6S },
	{{ "d_sfft32x32_c2r_perm2d_bias"     , "d_xfft32x32_c2r_perm2d_bias"     , "d_hfft32x32_c2r_perm2d_bias"      }, AM_4P_5S },
	{{ "d_sfft32x32_c2r_perm2d_bias_relu", "d_xfft32x32_c2r_perm2d_bias_relu", "d_hfft32x32_c2r_perm2d_bias_relu" }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm2d_bias_elu" , "d_xfft32x32_c2r_perm2d_bias_elu" , "d_hfft32x32_c2r_perm2d_bias_elu"  }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm2d_xdiff"    , "d_xfft32x32_c2r_perm2d_xdiff"    , "d_hfft32x32_c2r_perm2d_xdiff"     }, AM_4P_5S },
	{{ "d_sfft32x32_c2r_perm2d_brelu"    , "d_xfft32x32_c2r_perm2d_brelu"    , "d_hfft32x32_c2r_perm2d_brelu"     }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_perm2d_belu"     , "d_xfft32x32_c2r_perm2d_belu"     , "d_hfft32x32_c2r_perm2d_belu"      }, AM_4P_6S },
	{{ "d_sfft32x32_c2r_splice"          , "d_xfft32x32_c2r_splice"          , "d_hfft32x32_c2r_splice"           }, AM_3P_9S },
	{{ "d_sfft32x32_c2r_splice_frelu"    , "d_xfft32x32_c2r_splice_frelu"    , "d_hfft32x32_c2r_splice_frelu"     }, AM_3P_AS },
	{{ "d_sfft32x32_c2r_splice_felu"     , "d_xfft32x32_c2r_splice_felu"     , "d_hfft32x32_c2r_splice_felu"      }, AM_3P_AS },
	{{ "d_sfft32x32_c2r_splice_bias"     , "d_xfft32x32_c2r_splice_bias"     , "d_hfft32x32_c2r_splice_bias"      }, AM_4P_9S },
	{{ "d_sfft32x32_c2r_splice_bias_relu", "d_xfft32x32_c2r_splice_bias_relu", "d_hfft32x32_c2r_splice_bias_relu" }, AM_4P_AS },
	{{ "d_sfft32x32_c2r_splice_bias_elu" , "d_xfft32x32_c2r_splice_bias_elu" , "d_hfft32x32_c2r_splice_bias_elu"  }, AM_4P_AS },
	{{ "d_sfft32x32_c2r_splice_xdiff"    , "d_xfft32x32_c2r_splice_xdiff"    , "d_hfft32x32_c2r_splice_xdiff"     }, AM_4P_9S },
	{{ "d_sfft32x32_c2r_splice_brelu"    , "d_xfft32x32_c2r_splice_brelu"    , "d_hfft32x32_c2r_splice_brelu"     }, AM_4P_AS },
	{{ "d_sfft32x32_c2r_splice_belu"     , "d_xfft32x32_c2r_splice_belu"     , "d_hfft32x32_c2r_splice_belu"      }, AM_4P_AS },
	{{ "d_sfft32x32_c2r_filter"          , "d_xfft32x32_c2r_filter"          , "d_hfft32x32_c2r_filter"           }, AM_3P_5S }
};

void create_fft_kernel_r2c( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int i, int prc )
{
	cuda_context_create_kernel( p_kernel, p_ctx, fftkp_r2c[i].names[prc] );
	cuda_kernel_sao( p_kernel, fftkp_r2c[i].argmask );
	cuda_kernel_sbl( p_kernel, i<8?128:(i<12?32:64), (i<8)?1:8 );
	cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_global+g_fftRF_ofs[i>>2]*sizeof(float2) );
}
void create_fft_kernel_c2r( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int i, int prc )
{
	cuda_context_create_kernel( p_kernel, p_ctx, fftkp_c2r[i].names[prc] );
	cuda_kernel_sao( p_kernel, fftkp_c2r[i].argmask );
	cuda_kernel_sbl( p_kernel, i<20?128:(i<30?32:64), i<20?1:8 );
	cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_global+g_fftRF_ofs[i/10]*sizeof(float2) );
}
void create_cellfft_kernel_r2c( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int i, int prc )
{
	cuda_context_create_kernel( p_kernel, p_ctx, cellfftkp_r2c[i].names[prc] );
	cuda_kernel_sao( p_kernel, cellfftkp_r2c[i].argmask );
	cuda_kernel_sbl( p_kernel, i>7?512:256, 1 );
	cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_global+g_fftRF_ofs[i>7]*sizeof(float2) );
}
void create_cellfft_kernel_c2r( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int i, int prc )
{
	cuda_context_create_kernel( p_kernel, p_ctx, cellfftkp_c2r[i].names[prc] );
	cuda_kernel_sao( p_kernel, cellfftkp_c2r[i].argmask );
	cuda_kernel_sbl( p_kernel, 256, 1 );
	cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_global+g_fftRF_ofs[i>18]*sizeof(float2) );
}

