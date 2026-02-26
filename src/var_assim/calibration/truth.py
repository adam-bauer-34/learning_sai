from dataclasses import dataclass

@dataclass
class Truth:
    # WIP
    def from_cli_and_yaml(cls, clis, yaml):
        THETA_TR: float = clis.theta

        if clis.model != 'pco2geowc3':
            # setup two region model
            reg_fname = DATA_DIR / 'input' / 'regional_calibration_parameters.csv'
            df = pd.read_csv(reg_fname, delimiter=',', header=0, index_col='THETA')

            # central values
            # pattern scaling parameters for global temp
            self.ALPHA_R1_CEN = df.ALPHA_R1_CEN[int(THETA_TR)]
            self.ALPHA_R2_CEN = df.ALPHA_R2_CEN[int(THETA_TR)]

            # sai adjustment to pattern scaling
            self.BETA_R1_CEN = df.BETA_R1_CEN[int(THETA_TR)]
            self.BETA_R2_CEN = df.BETA_R2_CEN[int(THETA_TR)]

            # standard errors
            # pattern scaling parameters for global temp
            self.ALPHA_R1_STD = df.ALPHA_R1_STD[int(THETA_TR)]
            self.ALPHA_R2_STD = df.ALPHA_R2_STD[int(THETA_TR)]

            # sai adjustment to pattern scaling
            self.BETA_R1_STD = df.BETA_R1_STD[int(THETA_TR)]
            self.BETA_R2_STD = df.BETA_R2_STD[int(THETA_TR)]

        else:
            # setup three region model
            reg_fname = DATA_DIR / 'input' / 'regional_calibration_parameters_r3.csv'
            df = pd.read_csv(reg_fname, delimiter=',', header=0, index_col='THETA')

            # central values
            # pattern scaling parameters for global temp
            self.ALPHA_R1_CEN = df.ALPHA_R1_CEN[int(THETA_TR)]
            self.ALPHA_R2_CEN = df.ALPHA_R2_CEN[int(THETA_TR)]
            self.ALPHA_R3_CEN = df.ALPHA_R3_CEN[int(THETA_TR)]

            # sai adjustment to pattern scaling
            self.BETA_R1_CEN = df.BETA_R1_CEN[int(THETA_TR)]
            self.BETA_R2_CEN = df.BETA_R2_CEN[int(THETA_TR)]
            self.BETA_R3_CEN = df.BETA_R3_CEN[int(THETA_TR)]

            # standard errors
            # pattern scaling parameters for global temp
            self.ALPHA_R1_STD = df.ALPHA_R1_STD[int(THETA_TR)]
            self.ALPHA_R2_STD = df.ALPHA_R2_STD[int(THETA_TR)]
            self.ALPHA_R3_STD = df.ALPHA_R3_STD[int(THETA_TR)]

            # sai adjustment to pattern scaling
            self.BETA_R1_STD = df.BETA_R1_STD[int(THETA_TR)]
            self.BETA_R2_STD = df.BETA_R2_STD[int(THETA_TR)]
            self.BETA_R3_STD = df.BETA_R3_STD[int(THETA_TR)]