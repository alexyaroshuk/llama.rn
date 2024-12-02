# Resolve react_native_pods.rb with node to allow for hoisting
require Pod::Executable.execute_command('node', ['-p',
  'require.resolve(
    "react-native/scripts/react_native_pods.rb",
    {paths: [process.argv[1]]},
  )', __dir__]).strip

# Update minimum iOS version to 13.0 to support Metal features
platform :ios, '13.0'
prepare_react_native_project!

linkage = ENV['USE_FRAMEWORKS']
if linkage != nil
  Pod::UI.puts "Configuring Pod with #{linkage}ally linked Frameworks".green
  use_frameworks! :linkage => linkage.to_sym
end

target 'rntesttwo' do
  config = use_native_modules!

  use_react_native!(
    :path => config[:reactNativePath],
    :app_path => "#{Pod::Config.instance.installation_root}/.."
  )

  pod 'ReactCommon/turbomodule/core', :path => '../node_modules/react-native/ReactCommon'

  target 'rntesttwoTests' do
    inherit! :complete
  end

  post_install do |installer|
    # React Native post install
    react_native_post_install(
      installer,
      config[:reactNativePath],
      :mac_catalyst_enabled => false
    )

    # Fix RCTTurboModule import
    system('sed -i "" "s/#import <ReactCommon\\/RCTTurboModule.h>/#import \\"RCTTurboModule.h\\"/" "./build/generated/ios/RNLlamaSpec.h"')

    # Update deployment target for all pods
    installer.pods_project.targets.each do |target|
      target.build_configurations.each do |config|
        config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
      end
    end

    # Add Metal availability checks
    installer.pods_project.targets.each do |target|
      if target.name == 'llama-rn'
        target.build_configurations.each do |config|
          config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'] ||= ['$(inherited)']
          config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'] << 'GGML_METAL_SUPPORTS_FAMILY=1'
        end
      end
    end
  end
end

