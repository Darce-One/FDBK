// Dataset Generation

s.boot;

(
~p_all = 15000; // specify the number of generated samples
~duration = 0.19; // specify the duration of the generated samples
~sound_folder_name = "samples"; // dataset folder should be placed in the same folder as this file
~json_file_name = "params"; // params file will be created in the same folder as this file
)


// Init some stuff
(
s.waitForBoot{
	~paramsbuf = Buffer.loadCollection(s,[0, 0, 0, 0, 0, 0, 0, 0]);
	~ds_params = FluidDataSet(s);

	("fdbk_synthdef.scd").loadRelative;
	s.sync;

	~synthgen = Synth.new(\fdbk_train, [\dry_env, 1, \lag_t, 0, \delay_t, 0])}
)


// Generate random samples and save their parameters
(
fork{

	~p_all.do{
		arg i;
		var id = "sound-%".format(i);

		var fund, amp_ratio, fm_ratio, fm_index, pnoise_amp, lpf_freq, hpf_freq, osc_fm_ratio;

		fund = exprand(50.0, 2000.0);
		amp_ratio = rrand(0.5, 10.0);
		fm_ratio = rrand(0.25, 5.0);
		fm_index = rrand(0.0, 2.0);
		pnoise_amp = rrand(0.0, 1.0);
		lpf_freq = rrand(fund, 5000.0);
		hpf_freq = rrand(50.0, fund);
		osc_fm_ratio = rrand(0.05, 0.95);


		~paramsbuf.setn(0, [
			fund,
			amp_ratio,
			fm_ratio,
			fm_index,
			pnoise_amp,
			lpf_freq,
			hpf_freq,
			osc_fm_ratio
		]);

		~synthgen.set(
			\fund, fund,
			\amp_ratio, amp_ratio,
			\fm_ratio, fm_ratio,
			\fm_index, fm_index,
			\pnoise_amp, pnoise_amp,
			\lpf_freq, lpf_freq,
			\hpf_freq, hpf_freq,
			\osc_fm_ratio, osc_fm_ratio
		);
		s.sync;
		s.record(PathName(thisProcess.nowExecutingPath).parentPath +/+ ~sound_folder_name +/+ "sound-%.wav".format(i), duration: ~duration);
		~ds_params.addPoint(id, ~paramsbuf);
		(~duration + 0.05).wait;
		s.sync;
		s.stopRecording;
		s.sync;
		id.postln;
	};

	~ds_params.write(PathName(thisProcess.nowExecutingPath).parentPath +/+ "%.json".format(~json_file_name));

	~ds_params.print;
	~synthgen.free;
}
)
