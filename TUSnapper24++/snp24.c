/*
 *  $Id: snp24.c,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#include <asl_inc.h>

Terr EXPORT_FN
SNP24_read_video_data_raw(Thandle Hsnp24, void* Pdata, ui32 nbytes)
{
    static char		*fn_name = "SNP24_read_video_data_raw";
    struct Tsnp24	*Psnp24;
    Terr		ret_val;

    DPRINTF(3) ("\n%s: Entering", fn_name);

    SNP24_get_Psnp24(fn_name, Psnp24, Hsnp24);

    ret_val = Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_IS_BASE_MAPPER_FITTED);
    ASL_roe(ret_val);
    if (ASL_get_ret(ret_val) == 0)
    {
	SNP24_err_ret(Hsnp24, ASLERR_NOT_SUPPORTED, 0, 0, fn_name);
    }

    if (ASL_get_ret(SNP24_get_subsample(Hsnp24)) == SNP24_SUB_FAST)
    {
	SNP24_err_ret(Hsnp24, ASLERR_NOT_SUPPORTED, 0, 0, fn_name);
    }
    else if (SNP24_is_readout_deinterlace(Hsnp24) == TRUE &&
	     ASL_get_ret(SNP24_get_subsample(Hsnp24)) == SNP24_SUB_X1)
    {
	i16	roi[ASL_SIZE_2D_ROI];
	SNP24_get_ROI(Hsnp24, roi);
	
	/* Set automatic HBANK toggle to one image line width
	   - and tell SNP24 to accept HW toggle on HBANK */
	ASL_roe(Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_SET_DATAMAPPER,
			      MAPPER_SET_HBANK_COUNT, roi[2], (ui16) 0));
	ASL_roe(SNP24_set_read_bank(Hsnp24, SNP24_BANK_HW_DISABLE));
	/* Must HBANK_DISABLE before setting BANK_A */
	ASL_roe(SNP24_set_read_bank(Hsnp24, SNP24_BANK_READ_A));
	ASL_roe(SNP24_set_read_bank(Hsnp24,
				    (SNP24_BANK_HW_ENABLE |
				     SNP24_BANK_HW_HBANK_TOGGLE |
				     SNP24_BANK_READ_A)));
    }
    else
    {
	/* Turn off automatic HBANK toggle - always read one bank
	   - and tell SNP24 to ignore HW toggle & go back to A */
	ASL_roe(Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_SET_DATAMAPPER,
			      MAPPER_SET_HBANK_COUNT,
			      MAPPER_HBANK_DISABLE,
			      (ui16) 0));
	ASL_roe(SNP24_set_read_bank(Hsnp24, SNP24_BANK_HW_DISABLE));
	ASL_roe(SNP24_set_read_bank(Hsnp24, SNP24_BANK_READ_A));
    }

    ASL_roe(Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_INIT_VPORT));
    ASL_roe(Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_READ_VPORT_UI32,
			  Pdata, nbytes));
    ASL_roe(Psnp24->Pfunc(Psnp24->Hbaseboard, MODULE_RESET_VPORT));

    ASL_ok_ret(0);
}
