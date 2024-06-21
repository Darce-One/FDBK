// Dataset Generation


(
~p_all = 20000; // specify the number of generated samples
~duration = 0.19; // specify the duration of the generated samples
~sound_folder_name = "dataset"; // dataset folder should be placed in the same folder as this file
~json_file_name = "params"; // params file will be created in the same folder as this file
)


// Init some stuff
(
s.waitForBoot{
	~paramsbuf = Buffer.loadCollection(s,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
	~ds_params = FluidDataSet(s);

	("../perform/fdbk_synthdef.scd").loadRelative;
	s.sync;

	~synthgen = Synth.new(\fdbk, [\env_amount, 0, \lag_control, 0, \delay_control, 0])}
)


// Generate random samples and save their parameters
(
fork{

	~p_all.do{
		arg i;
		var id = "sound-%".format(i);

		var fund = exprand(50.0, 2000.0);
		var add_amp = rrand(0.0, 1.0);
		var blip_amp = rrand(0.0, 1.0);
		var fm_amp = rrand(0.0, 1.0);
		var ratio = exprand(0.01, 5.0);
		var index = rrand(0.0, 2.0);
		var pnoise_amp = rrand(0.0, 1.0);
		var bnoise_amp = rrand(0.0, 1.0);
		var lpf_freq = exprand(50.0, 5000.0);
		var hpf_freq = exprand(50.0, 5000.0);


		~paramsbuf.setn(0, [
			fund,
			add_amp,
			blip_amp,
			fm_amp,
			ratio,
			index,
			pnoise_amp,
			bnoise_amp,
			lpf_freq,
			hpf_freq
		]);

		~synthgen.set(
			\fund, fund,
			\add_amp, add_amp,
			\blip_amp, blip_amp,
			\fm_amp, fm_amp,
			\ratio, ratio,
			\index, index,
			\pnoise_amp, pnoise_amp,
			\bnoise_amp, bnoise_amp,
			\lpf_freq, lpf_freq,
			\hpf_freq, hpf_freq
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
