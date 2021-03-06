Create Table StateMeasures(     
				st_no numeric(500) not null,
                                st_name varchar(500) not null,
                                state char(2) primary key,
				s_hbips_2_measure_description varchar(500),
				s_hbips_2_overall_rate_per_1000 numeric(500),
				s_hbips_2_overall_num numeric(10) ,
				s_hbips_2_overall_den numeric(10) ,
				s_hbips_3_measure_description varchar(500),
				s_hbips_3_overall_rate_per_1000 numeric(500),
				s_hbips_3_overall_num numeric(10) ,
				s_hbips_3_overall_den numeric(10) ,
				s_hbips_5_measure_description varchar(500),
				s_hbips_5_overall_percentage_of_total numeric(5),
				s_hbips_5_overall_num numeric(10) ,
				s_hbips_5_overall_den numeric(10) ,
				s_hbips_6_measure_description varchar(500),
				s_hbips_6_overall_percentage_of_total numeric(5),
				s_hbips_6_overall_num numeric(10) ,
				s_hbips_6_overall_den numeric(10) ,
				s_hbips_7_measure_description varchar(500),
				s_hbips_7_overall_percentage_of_total numeric(5),
				s_hbips_7_overall_num numeric(10) ,
				s_hbips_7_overall_den numeric(10) ,
				s_sub_1_measure_description varchar(500),
				s_sub_1_percentage numeric(5),
				s_sub_1_numerator numeric(10) ,
				s_sub_1_denominator numeric(10) ,
				s_tob_1_measure_description varchar(500),
				s_tob_1_percentage numeric(5),
				s_tob_1_numerator numeric(10) ,
				s_tob_1_denominator numeric(10) ,
				tob_2_2a_measure_desc varchar(500),
				s_tob_2_percentage numeric(5),
				s_tob_2_numerator numeric(10) ,
				s_tob_2_2a_denominator numeric(10) ,	
				s_tob_2a_percentage numeric(5),
				s_tob_2a_numerator numeric(10) ,
				s_tob_2_2a__denominator numeric(10) ,
				s_peoc_measure_description varchar(500),
				s_peoc_yes_count int,
				s_peoc_no_count int,
				s_peoc_yes_percentage numeric(5),
				s_peoc_no_percentage numeric(5),
				s_ehr_use_measure_description varchar(500),
				s_ehr_paper_count int,
				s_ehr_non_certified_count int,
				s_ehr_certified_count int,	
				s_ehr_paper_percentage numeric(5),
				s_ehr_non_certified_percentage numeric(5),
				s_ehr_certified_percentage numeric(5),
				s_hie_measure_description varchar(500),
				s_hie_yes_count int,
				s_hie_no_count int,
				s_hie_yes_percent numeric(500),	
				s_hie_no_percent numeric(500),
				start_date date,
				end_date date,
				s_fuh_measure_description varchar(500),
				s_fuh_30_percentage numeric(5),	
				s_fuh_30_numerator numeric(10) ,
				s_fuh_30_denominator numeric(10) ,
				s_fuh_7_percentage numeric(5),
				s_fuh_7_numerator numeric(10) ,
				s_fuh_7_denominator numeric(10) ,
				s_fuh_measure_start_date date,
				s_fuh_measure_end_date date,
				s_imm_2_measure_description varchar(500),	
				s_imm_2_percentage numeric(5),
				s_imm_2_numerator numeric(10) ,
				s_imm_2_denominator numeric(10) ,
				s_hcp_measure_description varchar(500),
				s_hcp_percentage numeric(5),
				s_hcp_numerator numeric(10) ,	
				s_hcp_denominator numeric(10) ,
				s_flu_season_start_date date,	
				s_flu_season_end_date date);
